import tensorflow as tf
import transformers as t
import math
import tensorflow_models as tfm
from models_src.Support import download_mlflow_weights

def ResNet4Classification(input_shape=(256,256,3), FN_filters=[512], dropout=0.2, output_classes=4, config_args={}, name='ResNet', output_hidden_states=False):
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Permute((3,1,2))(inputs)
    x = t.TFResNetModel(config=t.ResNetConfig(**config_args), name='ResNet-core')(x, output_hidden_states=output_hidden_states)
    if output_hidden_states:
        outputs = x['hidden_states']
    else:
        x = tf.keras.layers.Flatten()(x['pooler_output'])
        x = tf.keras.layers.Dropout(dropout)(x)
        for filters in FN_filters:
            x = tf.keras.layers.Dense(filters, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    return model



class FeaturePyramid(tf.keras.Model):
    def __init__(self, out_indices=[1,2,3,4], output_dim=256, add_maxpool=True, **kwargs):
        super(FeaturePyramid, self).__init__(**kwargs)
        
        self.out_indices = out_indices
        self.out_permutes = [tf.keras.layers.Permute((2,3,1)) for idx in out_indices]

        self.equalize_channels_convs = [tf.keras.layers.Conv2D(output_dim, kernel_size=1) for idx in out_indices]
        self.upsamples = [tf.keras.layers.UpSampling2D(size=(2,2)) for idx in out_indices[::-1][:-1]]
        self.output_convs = [tf.keras.layers.Conv2D(output_dim, kernel_size=3, padding='same') for idx in out_indices[::-1]]

        self.add_maxpool = add_maxpool
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')

    def call(self, inputs, training=None):

        # get defined output hidden states from ResNet and permute back to [B H/i W/i C*i]
        hidden_states = [out_permute(inputs[idx]) for out_permute, idx in zip(self.out_permutes, self.out_indices)]

        # equalize channels number of every hidden state
        hidden_states = [conv(state) for state, conv in zip(hidden_states, self.equalize_channels_convs)][::-1]

        # add upsampled states from higher stage
        hidden_states = [hidden_states[0]] + [upsample(higher) + lower for upsample, higher, lower in zip(self.upsamples, hidden_states[:-1], hidden_states[1:])]

        # output convolutions
        hidden_states = [conv(state) for state, conv in zip(hidden_states, self.output_convs)]

        # add extra pooled state if defined
        if self.add_maxpool:
            hidden_states = [self.maxpool(hidden_states[0])] + hidden_states

        return hidden_states[::-1]
    



def gen_anchors(shape, anchor_size, window_size, anchor_scales, base_img_size):
    H, W = shape[1], shape[2]
    windows_num = math.ceil(H/window_size)*math.ceil(W/window_size)
    height, width = base_img_size
    anchors = len(anchor_scales)

    rows = tf.repeat(tf.reshape(tf.range(height, delta=height//H, dtype=tf.float32), (1,H,1,1)), W, axis=2)
    cols = tf.repeat(tf.reshape(tf.range(width, delta=width//W, dtype=tf.float32), (1,1,W,1)), H, axis=1)
    grid = tf.concat([cols, rows], axis=-1)

    pool = tf.keras.layers.AveragePooling2D(pool_size=window_size, padding='same')

    centers = tf.reshape(tf.reshape(tf.repeat(tf.expand_dims(pool(grid), axis=-2), anchors, axis=-2), (1,windows_num,2*anchors)), (1,windows_num*anchors,2))

    sizes = tf.stack([tf.constant([anchor_size/scale, anchor_size*scale], dtype=tf.float32) for scale in anchor_scales], axis=0) # [3,2]
    sizes =  tf.reshape(tf.repeat(sizes[tf.newaxis, tf.newaxis], windows_num, axis=1), (1, windows_num*anchors, 2))

    return tf.concat([centers, sizes], axis=-1)

class RegionProposalNetwork(tf.keras.layers.Layer):
    def __init__(self, 
                 anchor_sizes=[24,48,64,156,224],
                 anchor_scales=[0.5,1.0,2.0], 
                 window_sizes=[1,1,1,1,1], 
                 input_mapping=[0,1,2,3,4],
                 base_img_size=(256,256), 
                 init_top_k_proposals=0.3,
                 output_proposals=1000,
                 iou_threshold = 0.5,
                 add_bbox_dense_layer = False,
                 normalize_bboxes = False,
                 delta_scaler = [1.0,1.0,1.0,1.0],
                 bbox_training=True,
                 **kwargs):
        super(RegionProposalNetwork, self).__init__(**kwargs)

        self.anchors = len(anchor_scales)
        self.anchor_scales = anchor_scales
        self.anchor_sizes = anchor_sizes
        self.window_sizes = window_sizes
        self.base_img_size = base_img_size
        self.height, self.width = base_img_size
        self.input_mapping = input_mapping
        self.bbox_training = bbox_training

        self.img_size_tensor = tf.constant(base_img_size, dtype=tf.float32)[tf.newaxis, tf.newaxis]

        self.init_top_k_ratio = init_top_k_proposals
        self.output_proposals = output_proposals
        self.iou_threshold = iou_threshold

        self.normalize_bboxes = normalize_bboxes
        self.bbox_normalization = tf.constant(base_img_size*2, dtype=tf.float32)[tf.newaxis,tf.newaxis]
        self.add_bbox_dense_layer = add_bbox_dense_layer

        self.delta_scaler = tf.constant(delta_scaler, tf.float32)
    
    def _map_input(self, x):
        return [x[i] for i in self.input_mapping]

    def _bbox_decoding(self, bbox, anchor_bboxes):
        bbox *= self.delta_scaler
        dYX, dHW = bbox[...,:2], bbox[...,2:]
        YX, HW = anchor_bboxes[...,:2], anchor_bboxes[...,2:]

        YX += dYX*HW
        HW *= tf.math.exp(dHW)*0.5

        return tf.concat([tf.clip_by_value(YX-HW, [0,0], [self.height, self.width]), tf.clip_by_value(YX+HW, [0,0], [self.height, self.width])], axis=-1)
    
    def _top_k(self, confidence, bboxes, k):
        idxs = tf.math.top_k(confidence, k).indices
        confidence = tf.gather(confidence, idxs, batch_dims=1, axis=-1)
        bboxes = tf.gather(bboxes, idxs, batch_dims=1, axis=-2)

        return confidence, bboxes
    
    def _non_max_suppresion(self, confidence, bboxes):
        NMS_indices = tf.image.non_max_suppression(bboxes, confidence, max_output_size=self.output_proposals, iou_threshold=self.iou_threshold)

        last_idx = tf.reduce_max(tf.concat([NMS_indices, tf.constant([0])], axis=0))
        additional_proposals = self.output_proposals-len(NMS_indices)
        starting_point = tf.reduce_min(tf.stack([self.init_top_k-additional_proposals, last_idx+1], axis=0))

        NMS_bboxes = tf.gather(bboxes, NMS_indices, axis=0)
        additional_bboxes = bboxes[starting_point:(starting_point+additional_proposals)]

        NMS_confidence = tf.gather(confidence, NMS_indices, axis=0)
        additional_confidences = confidence[starting_point:(starting_point+additional_proposals)]

        confidence =  tf.concat([NMS_confidence, additional_confidences], axis=0)
        bboxes = tf.concat([NMS_bboxes, additional_bboxes], axis=0)

        confidence.set_shape((self.output_proposals,))
        bboxes.set_shape((self.output_proposals, 4))

        return confidence, bboxes
        

    def build(self, input_shape):
        input_shape = self._map_input(input_shape)

        self.in_convs = [tf.keras.layers.Conv2D(shape[-1], kernel_size=3, activation='relu', padding='same') for shape in input_shape]
        self.bbox_convs = [tf.keras.layers.Conv2D(self.anchors*4, kernel_size=window_size, strides=window_size, padding='same', kernel_initializer='zeros') for window_size in self.window_sizes]
        if self.add_bbox_dense_layer:
            self.bbox_dense = [tf.keras.layers.Dense(self.anchors*4, kernel_initializer='zeros') for _ in input_shape]
            if not self.bbox_training:
                for layer in self.bbox_dense:
                    layer.trainable = False

        if not self.bbox_training:
            for layer in self.bbox_convs:
                layer.trainable = False

        self.confidence_convs = [tf.keras.layers.Conv2D(self.anchors, kernel_size=window_size, strides=window_size, padding='same') for window_size in self.window_sizes]

        windows_nums = [math.ceil(shape[1]/window_size)*math.ceil(shape[2]/window_size) for shape, window_size in zip(input_shape, self.window_sizes)]
        print(f'windows nums: {windows_nums}')
        self.bbox_reshapes = [tf.keras.layers.Reshape((anchor_num*self.anchors,4)) for anchor_num in windows_nums]
        self.confidence_reshapes = [tf.keras.layers.Reshape((anchor_num*self.anchors,)) for anchor_num in windows_nums]

        self.sigmoids = [tf.keras.layers.Activation('sigmoid') for _ in input_shape]

        self.concat_confidence = tf.keras.layers.Concatenate(axis=-1)
        self.concat_bbox = tf.keras.layers.Concatenate(axis=-2)

        # generate anchor points
        self.anchor_bboxes = [gen_anchors(shape, anchor_size, window_size, self.anchor_scales, self.base_img_size) for 
                              shape, windows_num, anchor_size, window_size in zip(input_shape, windows_nums, self.anchor_sizes, self.window_sizes)]
        #self.anchor_sizes = [self._get_anchor_sizes(shape) for shape in input_shape]

        # initial proposal limit
        anchors_num = sum(windows_nums)*self.anchors
        self.init_top_k = int(anchors_num*self.init_top_k_ratio)
        print(f'all anchors num: {anchors_num}')
        print(f'top k anchors num: {self.init_top_k}')

    def call(self, inputs, training=None):
        inputs = self._map_input(inputs)
        # initial convolution for each feature in pyramid
        features = [conv(state) for state, conv in zip(inputs, self.in_convs)]

        # BBox confidence
        confidence = [sigmoid(reshape(conv(state))) for state, conv, reshape, sigmoid in zip(features, self.confidence_convs, self.confidence_reshapes, self.sigmoids)]

        # BBox predictions
        if self.add_bbox_dense_layer:
            bboxes = [reshape(dense(conv(state))) for state, conv, reshape, dense in zip(features, self.bbox_convs, self.bbox_reshapes, self.bbox_dense)]
        else:
            bboxes = [reshape(conv(state)) for state, conv, reshape in zip(features, self.bbox_convs, self.bbox_reshapes)]

        # Decode BBox predictions to original size and output format XYXY - left-top & right-bot
        bboxes = [self._bbox_decoding(b, a_b) for b, a_b in zip(bboxes, self.anchor_bboxes)]

        # concatenate standardized features
        confidence = self.concat_confidence(confidence)
        bboxes = self.concat_bbox(bboxes)

        # limit proposals to k with best scores
        confidence, bboxes = self._top_k(confidence, bboxes, k=self.init_top_k)

        # proceed nom max suppression
        confidence, bboxes = tf.map_fn(lambda x: self._non_max_suppresion(*x), elems=[confidence, bboxes], fn_output_signature=(tf.float32, tf.float32))

        if self.normalize_bboxes:
            bboxes /= self.bbox_normalization

        return confidence, bboxes
    
    def compute_output_shape(self, input_shape):
        batch_size = input_shape.as_list()[0]
        return ((batch_size, self.output_proposals), (batch_size, self.output_porposals, 4))
    

class ROIAligner(tf.keras.layers.Layer):
    def __init__(self, 
                 crop_size, 
                 sample_offset=0.5, 
                 source_indices=[1,2,3,4], 
                 pyramid_indices=[0,1,2,3],
                 **kwargs):
        super(ROIAligner, self).__init__(**kwargs)
        
        self.roi_align = tfm.vision.layers.MultilevelROIAligner(crop_size=crop_size, sample_offset=sample_offset)

        self.source_indices = source_indices
        self.pyramid_indices = pyramid_indices

    def call(self, feature_pyramid, bboxes, training=None):
        features = dict(map(lambda i,f: (str(i), feature_pyramid[f]), self.source_indices, self.pyramid_indices))
        rois = self.roi_align(features, bboxes)

        return rois
    

class CombinedMetricsModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.custom_metrics = [tf.keras.metrics.Mean(name='loss')]
    
    def add_metrics(self, metrics):
        '''
            metrics: list of dicts with obligatory keys:
                - metric - function of metric
                - label - model output key
                - weight_label - label of y_true_matched to use to sample_weight
        '''
        for metric_def in metrics:
            metric = metric_def['metric']
            metric.label = metric_def['label']
            metric.weight_label = metric_def['weight_label']
            self.custom_metrics += [metric]

    @property
    def metrics(self):
        return self.custom_metrics

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss, y_true_matched, y_pred_matched = self.compute_loss(y=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                weight_label = metric.weight_label
                sample_weight = y_true_matched[weight_label] if weight_label else None
                metric.update_state(y_true_matched[metric.label], y_pred_matched[metric.label], sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def compute_loss(self, y=None, y_pred=None):
        return self.loss(y, y_pred)
    

class ProposalPooling(tf.keras.layers.Layer):
    def __init__(self, reduction='max', **kwargs):
        super(ProposalPooling, self).__init__(**kwargs)

        self.reduction = tf.reduce_max if reduction=='max' else tf.reduce_mean

    def build(self, input_shape):
        P, H, W, C = input_shape[1], input_shape[2], input_shape[3], input_shape[4]

        self.perm = tf.keras.layers.Permute((1,4,2,3))
        self.flatten = tf.keras.layers.Reshape((P,C,H*W))

    def call(self, inputs):
        x = self.perm(inputs)
        x = self.flatten(x)
        x = self.reduction(x, axis=-1)

        return x

class FPNBBoxHead(tf.keras.layers.Layer):
    def __init__(self, 
                roi_size=7, 
                roi_source_indices=[1,2,3,4], 
                roi_pyramid_indices=[0,1,2,3], 
                hidden_sizes=1024, 
                hidden_layers=2, 
                pooling='max', 
                dropout=0.2, 
                **kwargs):
        super(FPNBBoxHead, self).__init__(**kwargs)

        self.roi_aligner = ROIAligner(crop_size=roi_size, source_indices=roi_source_indices, pyramid_indices=roi_pyramid_indices)
        self.pool = ProposalPooling(reduction=pooling)

        self.out_dropout, self.init_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(2)]
        self.dense_layers = tf.keras.Sequential([tf.keras.layers.Dense(hidden_sizes, activation='relu') for _ in range(hidden_layers)])

        self.class_predictor = tf.keras.layers.Dense(1, activation='sigmoid')
        self.class_flatten = tf.keras.layers.Flatten()
        self.bbox_predictor = tf.keras.layers.Dense(4, activation='sigmoid')

    def decode_bbox(self, bbox):
        YX, HW = bbox[...,:2], bbox[...,2:]/2
        bbox = tf.concat([YX-HW, YX+HW], axis=-1)
        return tf.clip_by_value(bbox, 0.0, 1.0)

    def call(self, pyramid, bboxes, training=None):
        x = self.roi_aligner(pyramid, bboxes)
        x = self.pool(x)
        x = self.init_dropout(x, training=training)
        x = self.dense_layers(x)

        confidence = self.class_flatten(self.class_predictor(x))
        bbox = self.bbox_predictor(x)
        bbox = self.decode_bbox(bbox)

        return confidence, bbox
    

    
class MaskHead(tf.keras.layers.Layer):
    def __init__(self,
                roi_size=32, 
                roi_source_indices=[1,2,3,4], 
                roi_pyramid_indices=[0,1,2,3],
                filters=256,
                kernel_size=3,
                output_size=(256,256),
                convs_num=4,
                upscale_blocks=3,
                upscale_block_convs=1,
                **kwargs
                ):
        super(MaskHead, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.roi_size = roi_size
        self.upscale_blocks = upscale_blocks

        self.roi_aligner = ROIAligner(crop_size=roi_size, source_indices=roi_source_indices, pyramid_indices=roi_pyramid_indices)

        self.convs = tf.keras.Sequential([tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu') for _ in range(convs_num)])
        self.upscales = tf.keras.Sequential([self._gen_upscale_block(upscale_block_convs) for _ in range(upscale_blocks)])

        self.out_conv = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid')

        self.resize = tf.keras.Sequential([tf.keras.layers.Permute((2,3,1)),
                                           tf.keras.layers.Resizing(*output_size)])

    def _gen_upscale_block(self, convs):
        return tf.keras.Sequential([tf.keras.layers.Conv2DTranspose(self.filters, self.kernel_size, strides=2, padding='same', activation='relu')]+
                                   [tf.keras.layers.Conv2D(self.filters, self.kernel_size, activation='relu', padding='same') for _ in range(convs)])
    
    def build(self, input_shape):
        self.P = input_shape[1][1]
        self.init_channels = input_shape[0][0][3]
        self.upscaled_size = self.roi_size*(2**self.upscale_blocks)

    def call(self, inputs, training=None):

        x = self.roi_aligner(*inputs) # [B,P,H,W,C(init)]
        x = tf.reshape(x, (-1, self.roi_size, self.roi_size, self.init_channels)) #[B*P,H,W,C(init)]
        x = self.convs(x) # [B*P,H,W,C]
        x = self.upscales(x) # [B*P, H*n, W*n, C]

        x = self.out_conv(x) # [B*P, H*n, W*n, 1]
        x = tf.reshape(x, (-1, self.P, self.upscaled_size, self.upscaled_size)) # [B,P,H*n,W*n]

        x = self.resize(x) # [B,H(out),W(out),P]
        return x
    

class MaskRCNNGenerator:
    def __init__(self, 
                 input_shape=(256,256,3), 
                 backbone_training=False, 
                 backbone_args={'config_args': {'hidden_sizes': [256,512,1024,2048]}, 'FN_filters': [512,512], 'output_hidden_states': True},
                 backbone_source={'experiment_id': '1030268364128310', 'run_name': 'languid-quail'},
                 backbone_download_weights=False,
                 backbone_pretrained=True,
                 RPN_training=False,
                 RPN_pyramid_args={'out_indices': [1,2,3,4], 'add_maxpool': True, 'output_dim': 256},
                 RPN_rpn_args={'anchor_sizes': [24,48,64,156,224], 'anchor_scales': [0.5,1.0,2.0], 'window_size': 1, 'init_top_k_proposals': 0.3, 'output_proposals': 2000, 'normalize_bboxes': True},
                 RPN_source={'experiment_id': '1520455876928689', 'run_name': 'monumental-crab'},
                 RPN_download_weights=False,
                 RPN_pretrained=False,
                 Head_training=False,
                 Head_download_weights=False,
                 Head_pretrained=False,
                 Head_pyramid_args={'roi_source_indices': [1,2,3,4], 'roi_pyramid_indices': [0,1,2,3]},
                 Head_BBox_args={'roi_size': 7, 'hidden_sizes': 1024, 'hidden_layers': 2},
                 Head_Mask_args={'roi_size': 32, 'filters': 256, 'kernel_size': 3, 'convs_num': 4, 'upscale_blocks': 3, 'upscale_block_convs': 1},
                 CombinedModel_metrics=None,
                 ):

        for key, value in locals().items():
            setattr(self, key, value)

        self.func_mapping = {
            'backbone': self._gen_backbone,
            'RPN': self._gen_RPN,
            'head': self._gen_Head
        }

        self.combined_model = False

    def _gen_backbone(self):
        if self.backbone_download_weights:
            download_mlflow_weights(**self.backbone_source, dst_path='./backbone')
            self.backbone_download_weights=False

        backbone = ResNet4Classification(input_shape=self.input_shape, **self.backbone_args)

        if self.backbone_pretrained:
            backbone.load_weights('./backbone/final_state/variables')

        backbone.trainable = self.backbone_training

        return backbone

    def _gen_RPN(self):
        if self.RPN_download_weights:
            download_mlflow_weights(**self.backbone_source, dst_path='./RPN')
            self.RPN_download_weights = False

        backbone = self._gen_backbone()

        inputs = tf.keras.layers.Input(self.input_shape)
        hidden_states = backbone(inputs)
        feature_pyramid = FeaturePyramid(name='FeaturePyramid', **self.RPN_pyramid_args, trainable=self.RPN_training)(hidden_states)
        rois_conf, rois = RegionProposalNetwork(name='RPN', **self.RPN_rpn_args, trainable=self.RPN_training)(feature_pyramid)

        if self.model_type=='RPN':
            RPN = CombinedMetricsModel(inputs, {'class': rois_conf, 'bbox':rois})
        else:
            RPN = tf.keras.Model(inputs, [feature_pyramid, rois], name='backbone-RPN')

        self.combined_model = True

        if self.RPN_pretrained:
            RPN.load_weights('./RPN/final_state/variables')

        return RPN
    
    def _gen_Head(self):
        if self.Head_download_weights:
            download_mlflow_weights(**self.Head_source, dst_path='./Head')
            self.Head_download_weights = False

        RPN = self._gen_RPN()

        inputs = tf.keras.layers.Input(self.input_shape)
        pyramid, bbox_proposals = RPN(inputs)
        conf, bboxes = FPNBBoxHead(name='BBoxHead', **self.Head_BBox_args, **self.Head_pyramid_args)(pyramid, bbox_proposals)
        masks = MaskHead(name='MaskHead', **self.Head_Mask_args, **self.Head_pyramid_args)([pyramid, bbox_proposals])

        Head = CombinedMetricsModel(inputs, {'class': conf, 'bbox': bboxes, 'mask': masks})

        self.combined_model = True

        if self.Head_pretrained:
            Head.load_weights('./Head/final_state/variables')

        return Head

    
    def __call__(self, model_type, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model_type = model_type
        model = self.func_mapping[model_type]()

        if (self.CombinedModel_metrics!=None) & self.combined_model:
            model.add_metrics(self.CombinedModel_metrics)

        return model