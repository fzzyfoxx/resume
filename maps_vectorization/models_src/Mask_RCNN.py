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
    
class RegionProposalNetwork(tf.keras.layers.Layer):
    def __init__(self, 
                 anchors=3, 
                 window_size=1, 
                 base_img_size=(256,256), 
                 init_top_k_proposals=0.3,
                 output_proposals=1000,
                 iou_threshold = 0.5,
                 normalize_bboxes = False,
                 **kwargs):
        super(RegionProposalNetwork, self).__init__(**kwargs)

        self.anchors = anchors
        self.window_size = window_size
        self.height, self.width = base_img_size

        self.img_size_tensor = tf.constant(base_img_size, dtype=tf.float32)[tf.newaxis, tf.newaxis]

        self.init_top_k_ratio = init_top_k_proposals
        self.output_proposals = output_proposals
        self.iou_threshold = iou_threshold

        self.normalize_bboxes = normalize_bboxes
        self.bbox_normalization = tf.constant(base_img_size*2, dtype=tf.float32)[tf.newaxis,tf.newaxis]

    def _get_anchor_centers(self, shape, anchor_num):
        H, W = shape[1], shape[2]
        rows = tf.repeat(tf.reshape(tf.range(self.height, delta=self.height//H, dtype=tf.float32), (1,H,1,1)), W, axis=2)
        cols = tf.repeat(tf.reshape(tf.range(self.width, delta=self.width//W, dtype=tf.float32), (1,1,W,1)), H, axis=1)
        grid = tf.concat([cols, rows], axis=-1)

        pool = tf.keras.layers.AveragePooling2D(pool_size=self.window_size, padding='same')

        return tf.reshape(tf.reshape(tf.repeat(tf.expand_dims(pool(grid), axis=-2), self.anchors, axis=-2), (1,anchor_num,2*self.anchors)), (1,anchor_num*self.anchors,2))
    
    def _get_anchor_sizes(self, shape):
        H, W = shape[1], shape[2]
        anchor_height = self.height/H*self.window_size
        anchor_width = self.width/W*self.window_size

        return tf.cast([anchor_height, anchor_width], dtype=tf.float32)[tf.newaxis, tf.newaxis]

    def _bbox_decoding(self, bbox, anchor_centers, anchor_sizes):
        YX, HW = bbox[...,:2], bbox[...,2:]

        YX = YX*self.img_size_tensor+anchor_centers
        HW = (tf.math.exp(HW)*self.img_size_tensor+anchor_sizes)/2

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

        self.in_convs = [tf.keras.layers.Conv2D(shape[-1], kernel_size=3, activation='relu', padding='same') for shape in input_shape]
        self.bbox_convs = [tf.keras.layers.Conv2D(self.anchors*4, kernel_size=self.window_size, strides=self.window_size, padding='same') for _ in input_shape]
        self.confidence_convs = [tf.keras.layers.Conv2D(self.anchors, kernel_size=self.window_size, strides=self.window_size, padding='same') for _ in input_shape]

        windows_nums = [math.ceil(shape[1]/self.window_size)*math.ceil(shape[2]/self.window_size) for shape in input_shape]
        print(f'windows nums: {windows_nums}')
        self.bbox_reshapes = [tf.keras.layers.Reshape((anchor_num*self.anchors,4)) for anchor_num in windows_nums]
        self.confidence_reshapes = [tf.keras.layers.Reshape((anchor_num*self.anchors,)) for anchor_num in windows_nums]

        self.sigmoids = [tf.keras.layers.Activation('sigmoid') for _ in input_shape]

        self.concat_confidence = tf.keras.layers.Concatenate(axis=-1)
        self.concat_bbox = tf.keras.layers.Concatenate(axis=-2)

        # generate anchor points
        self.anchor_centers = [self._get_anchor_centers(shape, windows_num) for shape, windows_num in zip(input_shape, windows_nums)]
        self.anchor_sizes = [self._get_anchor_sizes(shape) for shape in input_shape]

        # initial proposal limit
        anchors_num = sum(windows_nums)*self.anchors
        self.init_top_k = int(anchors_num*self.init_top_k_ratio)
        print(f'all anchors num: {anchors_num}')
        print(f'top k anchors num: {self.init_top_k}')

    def call(self, inputs, training=None):
        # initial convolution for each feature in pyramid
        features = [conv(state) for state, conv in zip(inputs, self.in_convs)]

        # BBox confidence
        confidence = [sigmoid(reshape(conv(state))) for state, conv, reshape, sigmoid in zip(features, self.confidence_convs, self.confidence_reshapes, self.sigmoids)]

        # BBox predictions
        bboxes = [reshape(conv(state)) for state, conv, reshape in zip(features, self.bbox_convs, self.bbox_reshapes)]

        # Decode BBox predictions to original size and output format XYXY - left-top & right-bot
        bboxes = [self._bbox_decoding(b,a_c,a_s) for b,a_c,a_s in zip(bboxes, self.anchor_centers, self.anchor_sizes)]

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
                 RPN_rpn_args={'anchors': 3, 'window_size': 1, 'init_top_k_proposals': 0.3, 'output_proposals': 1000, 'normalize_bboxes': True},
                 RPN_ROI_args={'crop_size': 7, 'source_indices': [1,2,3,4]},
                 RPN_ROI_output=False,
                 RPN_source={'experiment_id': None, 'run_name': None},
                 RPN_download_weights=False,
                 RPN_pretrained=False,
                 CombinedModel_metrics=None,
                 ):

        for key, value in locals().items():
            setattr(self, key, value)

        self.func_mapping = {
            'backbone': self._gen_backbone,
            'RPN': self._gen_RPN
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
        
        if self.RPN_ROI_output:
            rois = ROIAligner(name='ROI', **self.RPN_ROI_args)(feature_pyramid, rois)
            RPN = tf.keras.Model(inputs, rois)
        else:
            RPN = CombinedMetricsModel(inputs, {'class': rois_conf, 'bbox':rois})
            self.combined_model = True

        if self.RPN_pretrained:
            RPN.load_weights('./RPN/final_state/variables')

        return RPN
    
    def __call__(self, model_type, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

        model = self.func_mapping[model_type]()

        if (self.CombinedModel_metrics!=None) & self.combined_model:
            model.add_metrics(self.CombinedModel_metrics)

        return model