import tensorflow as tf
import transformers as t
import math
#import tensorflow_models as tfm

from models_src.Trainer_support import download_mlflow_weights
from models_src.CombinedMetricsModel import CombinedMetricsModel
from models_src.Metrics import MultivariantHungarianLoss, IoU
    

def ResNet4Classification(input_shape=(256,256,3), FN_filters=[512], dropout=0.2, output_classes=4, config_args={}, name='ResNet', output_hidden_states=False):
    """Build a ResNet model for classification.

    Args:
        input_shape: tuple, input image shape.
        FN_filters: list of integers, fully connected layer filters.
        dropout: float, dropout rate.
        output_classes: integer, number of output classes.
        config_args: dict, configuration for ResNet backbone.
        name: string, model name.
        output_hidden_states: bool, whether to output hidden states.

    Returns:
        A tf.keras.Model instance.
    """
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
    """Feature Pyramid Network (FPN) layer wrapper.

    Builds a feature pyramid from backbone hidden states. The class equalizes
    channel dimensions, upsamples higher-level features and combines them with
    lower-level features, and optionally adds an extra max-pooled level.

    Attributes:
        out_indices: indices of backbone hidden states to use.
        output_dim: number of channels for each pyramid level.
        add_maxpool: whether to add an extra pooled level.
    """
    def __init__(self, out_indices=[1,2,3,4], output_dim=256, add_maxpool=True, **kwargs):
        """Initialize the FeaturePyramid.

        Args:
            out_indices: list of indices selecting backbone feature maps.
            output_dim: integer, channel dimension for pyramid feature maps.
            add_maxpool: bool, whether to prepend a maxpooled feature map.
            **kwargs: forwarded to tf.keras.Model.
        """
        super(FeaturePyramid, self).__init__(**kwargs)
        
        self.out_indices = out_indices
        self.out_permutes = [tf.keras.layers.Permute((2,3,1)) for idx in out_indices]

        self.equalize_channels_convs = [tf.keras.layers.Conv2D(output_dim, kernel_size=1) for idx in out_indices]
        self.upsamples = [tf.keras.layers.UpSampling2D(size=(2,2)) for idx in out_indices[::-1][:-1]]
        self.output_convs = [tf.keras.layers.Conv2D(output_dim, kernel_size=3, padding='same') for idx in out_indices[::-1]]

        self.add_maxpool = add_maxpool
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')

    def call(self, inputs, training=None):
        """Run the forward pass.

        Args:
            inputs: list or tuple of backbone hidden states.
            training: bool or None, training flag forwarded to layers.

        Returns:
            List of feature maps for each pyramid level in ascending order
            (from low resolution to high resolution) matching out_indices.
        """
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
    """Generate anchor boxes for a given feature map shape.

    Args:
        shape: tuple, shape of the feature map.
        anchor_size: float, base anchor size.
        window_size: int, pooling window size.
        anchor_scales: list of floats, scales for anchor boxes.
        base_img_size: tuple, size of the base image.

    Returns:
        Tensor of anchor boxes with shape [1, num_anchors, 4].
    """
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


class NonMaxSupression(tf.keras.layers.Layer):
    """Non-maximum suppression helper layer.

    This layer selects top-k proposals by score and applies NMS to remove
    overlapping boxes, returning a fixed number of output proposals.
    """
    def __init__(self, 
                 init_top_k_proposals=0.5, 
                 iou_threshold=0.5, 
                 output_proposals=200,
                 split_input=False,
                 **kwargs):
        """Initialize NMS layer.

        Args:
            init_top_k_proposals: float or ratio of initial proposals to keep
                before NMS (interpreted relative to total anchors).
            iou_threshold: float IoU threshold for suppression.
            output_proposals: int number of proposals to return.
            split_input: bool, whether inputs are provided split (confidence, bboxes).
            **kwargs: forwarded to base Layer.
        """
        super(NonMaxSupression, self).__init__(**kwargs)

        self.init_top_k_ratio = init_top_k_proposals
        self.output_proposals = output_proposals
        self.iou_threshold = iou_threshold
        self.split = split_input

    def _top_k(self, confidence, bboxes, k):
        """Select top-k proposals by score per batch element.

        Args:
            confidence: tensor shape [B, A] of scores.
            bboxes: tensor shape [B, A, 4] of boxes.
            k: integer number of top proposals to select.

        Returns:
            (confidence_k, bboxes_k) tensors restricted to top-k indices.
        """
        idxs = tf.math.top_k(confidence, k).indices
        confidence = tf.gather(confidence, idxs, batch_dims=1, axis=-1)
        bboxes = tf.gather(bboxes, idxs, batch_dims=1, axis=-2)

        return confidence, bboxes

    def _non_max_suppresion(self, confidence, bboxes):
        """Apply TF non_max_suppression on a single example.

        Args:
            confidence: 1-D tensor of scores for one image.
            bboxes: 2-D tensor of boxes corresponding to confidence.

        Returns:
            (confidence_out, bboxes_out) trimmed/padded to output_proposals.
        """
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
        """Compute initial top-K based on input anchor count.

        Args:
            input_shape: shape(s) of inputs provided at build time.
        """
        if self.split:
            anchors_num = input_shape[1]
        else:
            anchors_num = input_shape[0][1]
        self.init_top_k = int(anchors_num*self.init_top_k_ratio)

    def call(self, inputs, training=None):
        """Execute selection and NMS on a batch of proposals.

        Args:
            inputs: either [confidence, bboxes] or a concatenated input when
                split is True.
            training: ignored, exists for Keras compatibility.

        Returns:
            Tuple of (confidence_selected, bboxes_selected).
        """
        if self.split:
            confidence, bboxes = inputs[...,0], inputs[...,1:]
        else:
            confidence, bboxes = inputs[0], inputs[1]
        # limit proposals to k with best scores
        confidence, bboxes = self._top_k(confidence, bboxes, k=self.init_top_k)

        # proceed nom max suppression
        confidence, bboxes = tf.map_fn(lambda x: self._non_max_suppresion(*x), elems=[confidence, bboxes], fn_output_signature=(tf.float32, tf.float32))

        return confidence, bboxes

    def compute_output_shape(self, input_shape):
        """Report output shapes for Keras compatibility.

        Args:
            input_shape: input shape passed to the layer.
        """
        batch_size = input_shape.as_list()[0]
        return ((batch_size, self.output_proposals), (batch_size, self.output_porposals, 4))
    


class RegionProposalNetwork(tf.keras.layers.Layer):
    """Region Proposal Network (RPN) layer.

    Generates objectness scores and bounding-box proposals from a feature
    pyramid. Supports decoding bbox deltas relative to predefined anchors,
    concatenation of proposals across pyramid levels and optional NMS.
    """
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
                 confidence_training=True,
                 nms=True,
                 **kwargs):
        """Initialize RPN.

        Args:
            anchor_sizes, anchor_scales, window_sizes: anchor generation params.
            input_mapping: mapping from pyramid outputs to RPN inputs.
            base_img_size: (H,W) of images used to generate anchors.
            init_top_k_proposals: ratio or fraction for initial top-k selection.
            output_proposals: number of proposals to output after NMS.
            iou_threshold: float IoU NMS threshold.
            add_bbox_dense_layer: whether to add dense bbox head.
            normalize_bboxes: whether to normalize bbox outputs to [0,1].
            delta_scaler: scaling applied to bbox deltas.
            bbox_training/confidence_training: booleans controlling trainable flags.
            nms: whether to run non-max suppression.
            **kwargs: forwarded to base Layer.
        """
        super(RegionProposalNetwork, self).__init__(**kwargs)

        self.anchors = len(anchor_scales)
        self.anchor_scales = anchor_scales
        self.anchor_sizes = anchor_sizes
        self.window_sizes = window_sizes
        self.base_img_size = base_img_size
        self.height, self.width = base_img_size
        self.input_mapping = input_mapping
        self.nms = nms

        self.bbox_training = bbox_training
        self.confidence_training = confidence_training

        self.img_size_tensor = tf.constant(base_img_size, dtype=tf.float32)[tf.newaxis, tf.newaxis]

        self.init_top_k_ratio = init_top_k_proposals
        self.output_proposals = output_proposals
        self.iou_threshold = iou_threshold

        self.normalize_bboxes = normalize_bboxes
        self.bbox_normalization = tf.constant(base_img_size*2, dtype=tf.float32)[tf.newaxis,tf.newaxis]
        self.add_bbox_dense_layer = add_bbox_dense_layer

        self.delta_scaler = tf.constant(delta_scaler, tf.float32)
    
    def _map_input(self, x):
        """Select mapped inputs from a list/tuple according to input_mapping.

        Args:
            x: list/tuple of tensors (feature maps).

        Returns:
            List of tensors selected using input_mapping.
        """
        return [x[i] for i in self.input_mapping]

    def _bbox_decoding(self, bbox, anchor_bboxes):
        """Decode bbox deltas to absolute box coordinates XYXY.

        Args:
            bbox: tensor of bbox deltas (...,4).
            anchor_bboxes: tensor of anchor centers and sizes (...,4).

        Returns:
            Decoded absolute bboxes clipped to image bounds in format [y1,x1,y2,x2].
        """
        bbox *= self.delta_scaler
        dYX, dHW = bbox[...,:2], bbox[...,2:]
        YX, HW = anchor_bboxes[...,:2], anchor_bboxes[...,2:]

        YX += dYX*HW
        HW *= tf.math.exp(dHW)*0.5

        return tf.concat([tf.clip_by_value(YX-HW, [0,0], [self.height, self.width]), tf.clip_by_value(YX+HW, [0,0], [self.height, self.width])], axis=-1)

    def build(self, input_shape):
        """Build internal layers based on input shapes.

        Args:
            input_shape: shapes of the incoming pyramid feature maps.
        """
        input_shape = self._map_input(input_shape)

        self.in_convs = [tf.keras.layers.Conv2D(shape[-1], kernel_size=3, activation='relu', padding='same', trainable=self.confidence_training) for shape in input_shape]

        self.confidence_convs = [tf.keras.layers.Conv2D(self.anchors, kernel_size=window_size, strides=window_size, padding='same', trainable=self.confidence_training) 
                                 for window_size in self.window_sizes]
        

        self.bbox_convs = [tf.keras.layers.Conv2D(self.anchors*4, kernel_size=window_size, strides=window_size, padding='same', 
                                                  kernel_initializer='zeros', trainable=self.bbox_training) 
                           for window_size in self.window_sizes]
        
        if self.add_bbox_dense_layer:
            self.bbox_dense = [tf.keras.layers.Dense(self.anchors*4, kernel_initializer='zeros', trainable=self.bbox_training) for _ in input_shape]


        

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
        if not self.nms:
            self.output_proposals = anchors_num
        #init_top_k = int(anchors_num*self.init_top_k_ratio)
        self.NMS = NonMaxSupression(self.init_top_k_ratio, self.iou_threshold, self.output_proposals)
        self.NMS.build([(None, anchors_num),(None, anchors_num, 4)])
        print(f'all anchors num: {anchors_num}')
        print(f'top k anchors num: {self.NMS.init_top_k}')

    def call(self, inputs, training=None):
        """Compute RPN confidences and decoded bounding boxes.

        Args:
            inputs: feature pyramid tensors as a list/tuple.
            training: bool, whether in training mode.

        Returns:
            Tuple (confidence, bboxes) where confidence shape is [B,P]
            and bboxes shape is [B,P,4] (optionally normalized).
        """
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

        if self.nms:
        # limit proposals to k with best scores
        # proceed nom max suppression
            confidence, bboxes = self.NMS([confidence, bboxes])

        if self.normalize_bboxes:
            bboxes /= self.bbox_normalization

        return confidence, bboxes
    
    def compute_output_shape(self, input_shape):
        """Report output shapes for Keras compatibility.

        Args:
            input_shape: input shape passed to the layer.
        """
        batch_size = input_shape.as_list()[0]
        return ((batch_size, self.output_proposals), (batch_size, self.output_porposals, 4))
    

class ROIAligner(tf.keras.layers.Layer):
    """Wrapper for multilevel ROI alignment.

    This class is a thin wrapper intended to call into a multilevel ROI aligner
    (e.g. from tensorflow_models). It handles mapping between pyramid and
    aligner inputs.
    """
    def __init__(self, 
                 crop_size, 
                 sample_offset=0.5, 
                 source_indices=[1,2,3,4], 
                 pyramid_indices=[0,1,2,3],
                 **kwargs):
        """Initialize ROIAligner.

        Args:
            crop_size: output size for each ROI crop (scalar).
            sample_offset: sampling offset for ROI aligner implementation.
            source_indices: indices in pyramid used as ROI sources.
            pyramid_indices: mapping indices for the underlying aligner.
            **kwargs: forwarded to base Layer.
        """
        super(ROIAligner, self).__init__(**kwargs)
        
        self.roi_align = None#tfm.vision.layers.MultilevelROIAligner(crop_size=crop_size, sample_offset=sample_offset)

        self.source_indices = source_indices
        self.pyramid_indices = pyramid_indices

    def call(self, feature_pyramid, bboxes, training=None):
        """Perform ROI alignment over the pyramid.

        Args:
            feature_pyramid: list of feature maps from FeaturePyramid.
            bboxes: boxes to crop for each image and proposal.
            training: bool, unused but accepted for compatibility.

        Returns:
            Tensor of aligned ROI crops as expected by downstream heads.
        """
        features = dict(map(lambda i,f: (str(i), feature_pyramid[f]), self.source_indices, self.pyramid_indices))
        rois = self.roi_align(features, bboxes)

        return rois

    

class CombinedMetricsModel(tf.keras.Model):
    """Keras Model subclass that supports combined custom metrics.

    This model keeps a list of custom metrics and uses custom train/test
    steps which rely on compute_loss returning (loss, y_true_matched,
    y_pred_matched). The class exposes helper methods to add and update
    metrics during training/testing.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the CombinedMetricsModel.

        Args:
            *args, **kwargs: forwarded to tf.keras.Model.
        """
        super(CombinedMetricsModel, self).__init__(*args, **kwargs)

        self.custom_metrics = [tf.keras.metrics.Mean(name='loss')]
    
    def add_metrics(self, metrics):
        """Register additional metrics to be tracked.

        Args:
            metrics: list of dicts containing keys 'metric', 'label', and
                'weight_label' describing how to compute each metric.
        """
        for metric_def in metrics:
            metric = metric_def['metric']
            metric.label = metric_def['label']
            metric.weight_label = metric_def['weight_label']
            self.custom_metrics += [metric]

    @property
    def metrics(self):
        """Return the list of registered metrics for Keras integration."""
        return self.custom_metrics
    
    def update_metrics(self, loss, y_true_matched, y_pred_matched):
        """Update internal Keras Metric objects with latest batch results.

        Args:
            loss: scalar or tensor loss for the current batch.
            y_true_matched: dict of matched ground-truth tensors for metrics.
            y_pred_matched: dict of matched prediction tensors for metrics.

        Returns:
            Dict mapping metric name to current value (metric.result()).
        """
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                weight_label = metric.weight_label
                sample_weight = y_true_matched[weight_label] if weight_label else None
                metric.update_state(y_true_matched[metric.label], y_pred_matched[metric.label], sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        """Custom training step integrating compute_loss and metrics.

        Args:
            data: tuple (x, y) passed by model.fit.

        Returns:
            Dict of metric names to values after applying gradients.
        """
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
        return self.update_metrics(loss, y_true_matched, y_pred_matched)
    
    def test_step(self, data):
        """Custom test/validation step used by model.evaluate.

        Args:
            data: tuple (x, y) passed by model.evaluate.

        Returns:
            Dict of metric names to values computed on the batch.
        """
        x, y = data
        y_pred = self(x, training=False)
        loss, y_true_matched, y_pred_matched = self.compute_loss(y=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)

        return self.update_metrics(loss, y_true_matched, y_pred_matched)

    def compute_loss(self, y=None, y_pred=None):
        """Delegate to configured loss callable.

        Args:
            y: ground-truth targets.
            y_pred: model predictions.

        Returns:
            Whatever the configured loss returns (loss, y_true_matched, y_pred_matched).
        """
        return self.loss(y, y_pred)
    
    def save(self, *args, **kwargs):
        """Save a Keras model built from current inputs/outputs.

        This wraps tf.keras.Model saving to persist the model structure and
        weights.
        """
        tf.keras.Model(self.input, self.output).save(*args, **kwargs)
    

class ProposalPooling(tf.keras.layers.Layer):
    """Pooling layer that aggregates per-proposal features.

    This layer flattens spatial dimensions for each proposal and applies
    a reduction (max or mean) to produce a compact descriptor per proposal.
    """
    def __init__(self, reduction='max', **kwargs):
        """Initialize ProposalPooling.

        Args:
            reduction: 'max' or any other value indicating mean pooling.
            **kwargs: forwarded to base Layer.
        """
        super(ProposalPooling, self).__init__(**kwargs)

        self.reduction = tf.reduce_max if reduction=='max' else tf.reduce_mean

    def build(self, input_shape):
        """Build internal permutation and reshaping layers based on input.

        Args:
            input_shape: expected input shape describing proposals.
        """
        P, H, W, C = input_shape[1], input_shape[2], input_shape[3], input_shape[4]

        self.perm = tf.keras.layers.Permute((1,4,2,3))
        self.flatten = tf.keras.layers.Reshape((P,C,H*W))

    def call(self, inputs):
        """Aggregate per-proposal spatial features.

        Args:
            inputs: tensor of shape [B, P, H, W, C]

        Returns:
            Tensor [B, P, C] representing pooled features per proposal.
        """
        x = self.perm(inputs)
        x = self.flatten(x)
        x = self.reduction(x, axis=-1)

        return x

class FPNBBoxHead(tf.keras.layers.Layer):
    """Bounding-box head operating on FPN-aligned proposal features.

    This head performs ROI alignment, pooling, fully-connected layers and
    predicts a class score and a refined bbox per proposal.
    """
    def __init__(self, 
                roi_size=7, 
                roi_source_indices=[1,2,3,4], 
                roi_pyramid_indices=[0,1,2,3], 
                hidden_sizes=1024, 
                hidden_layers=2, 
                pooling='max', 
                dropout=0.2, 
                **kwargs):
        """Initialize bbox head components.

        Args:
            roi_size: spatial size for ROI alignment.
            roi_source_indices/roi_pyramid_indices: mapping for ROIAligner.
            hidden_sizes/layers: FC configuration for classification/regression.
            pooling: pooling method for proposal aggregation.
            dropout: dropout rate for FC layers.
            **kwargs: forwarded to base Layer.
        """
        super(FPNBBoxHead, self).__init__(**kwargs)

        self.roi_aligner = ROIAligner(crop_size=roi_size, source_indices=roi_source_indices, pyramid_indices=roi_pyramid_indices)
        self.pool = ProposalPooling(reduction=pooling)

        self.out_dropout, self.init_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(2)]
        self.dense_layers = tf.keras.Sequential([tf.keras.layers.Dense(hidden_sizes, activation='relu') for _ in range(hidden_layers)])

        self.class_predictor = tf.keras.layers.Dense(1, activation='sigmoid')
        self.class_flatten = tf.keras.layers.Flatten()
        self.bbox_predictor = tf.keras.layers.Dense(4, activation='sigmoid')

    def decode_bbox(self, bbox):
        """Decode predicted bbox representation to XYXY normalized coords.

        Args:
            bbox: tensor of predicted [y_center,x_center,height,width] style values
                  where height/width are split and halved internally.

        Returns:
            Clipped bbox tensor in normalized [y1,x1,y2,x2] format.
        """
        YX, HW = bbox[...,:2], bbox[...,2:]/2
        bbox = tf.concat([YX-HW, YX+HW], axis=-1)
        return tf.clip_by_value(bbox, 0.0, 1.0)

    def call(self, pyramid, bboxes, training=None):
        """Forward pass through the bbox head.

        Args:
            pyramid: list of FPN feature maps.
            bboxes: proposals to refine.
            training: bool, whether in training mode.

        Returns:
            (confidence, bbox) where confidence is [B,P] and bbox is [B,P,4].
        """
        x = self.roi_aligner(pyramid, bboxes)
        x = self.pool(x)
        x = self.init_dropout(x, training=training)
        x = self.dense_layers(x)

        confidence = self.class_flatten(self.class_predictor(x))
        bbox = self.bbox_predictor(x)
        bbox = self.decode_bbox(bbox)

        return confidence, bbox
    

    
class MaskHead(tf.keras.layers.Layer):
    """Mask prediction head that generates instance masks from pooled ROIs.

    The head aligns ROIs, applies a small conv-stack, upsamples the feature
    maps and predicts per-proposal masks which are finally resized to the
    target output resolution.
    """
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
        """Initialize the MaskHead.

        Args:
            roi_size: input crop size for each ROI.
            filters/kernel_size: conv hyperparameters.
            output_size: final mask output image size (H,W).
            convs_num: number of conv layers before upsampling.
            upscale_blocks: number of upsampling stages.
            upscale_block_convs: additional convs per upsampling block.
            **kwargs: forwarded to base Layer.
        """
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
        """Create a single upsampling block.

        Args:
            convs: number of conv layers to follow the transposed conv.

        Returns:
            A tf.keras.Sequential representing an upsampling block.
        """
        return tf.keras.Sequential([tf.keras.layers.Conv2DTranspose(self.filters, self.kernel_size, strides=2, padding='same', activation='relu')]+
                                   [tf.keras.layers.Conv2D(self.filters, self.kernel_size, activation='relu', padding='same') for _ in range(convs)])
    
    def build(self, input_shape):
        """Determine proposal count and channel dims from inputs.

        Args:
            input_shape: shape info passed when build is triggered.
        """
        self.P = input_shape[1][1]
        self.init_channels = input_shape[0][0][3]
        self.upscaled_size = self.roi_size*(2**self.upscale_blocks)

    def call(self, inputs, training=None):
        """Predict instance masks for a batch of proposals.

        Args:
            inputs: tuple (pyramid, bboxes) as expected by ROIAligner.
            training: bool flag for dropout/other layers.

        Returns:
            Tensor of masks resized to output_size per image: [B,H_out,W_out,P].
        """
        x = self.roi_aligner(*inputs) # [B,P,H,W,C(init)]
        x = tf.reshape(x, (-1, self.roi_size, self.roi_size, self.init_channels)) #[B*P,H,W,C(init)]
        x = self.convs(x) # [B*P,H,W,C]
        x = self.upscales(x) # [B*P, H*n, W*n, C]

        x = self.out_conv(x) # [B*P, H*n, W*n, 1]
        x = tf.reshape(x, (-1, self.P, self.upscaled_size, self.upscaled_size)) # [B,P,H*n,W*n]

        x = self.resize(x) # [B,H(out),W(out),P]
        return x
    

class MaskRCNNGenerator:
    """Factory for assembling Mask R-CNN model components.

    This helper class constructs backbone, RPN and head components according
    to provided configurations and can load pretrained weights if requested.
    """
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
        """Construct a MaskRCNNGenerator storing provided configuration.

        The constructor saves all provided arguments as attributes to enable
        flexible overrides prior to calling the generator to build a model.
        """

        for key, value in locals().items():
            setattr(self, key, value)

        self.func_mapping = {
            'backbone': self._gen_backbone,
            'RPN': self._gen_RPN,
            'head': self._gen_Head
        }

        self.combined_model = False

    def _gen_backbone(self):
        """Build or load the backbone model according to configuration.

        Returns:
            A tf.keras.Model backbone instance.
        """
        if self.backbone_download_weights:
            download_mlflow_weights(**self.backbone_source, dst_path='./backbone')
            self.backbone_download_weights=False

        backbone = ResNet4Classification(input_shape=self.input_shape, **self.backbone_args)

        if self.backbone_pretrained:
            backbone.load_weights('./backbone/final_state/variables')

        backbone.trainable = self.backbone_training

        return backbone

    def _gen_RPN(self):
        """Assemble backbone + FeaturePyramid + RegionProposalNetwork.

        Returns:
            If model_type is 'RPN' a CombinedMetricsModel is returned; otherwise
            a tf.keras.Model returning the pyramid and proposals is returned.
        """
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
        """Assemble full detection head (bbox + mask) connected to RPN output.

        Returns:
            A CombinedMetricsModel representing the head that predicts class,
            bbox and mask outputs.
        """
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
        """Generate model specified by model_type using stored configuration.

        Args:
            model_type: string key in ['backbone','RPN','head'] to indicate
                which model to construct.
            **kwargs: overrides for stored configuration attributes.

        Returns:
            Constructed Keras model.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model_type = model_type
        model = self.func_mapping[model_type]()

        if (self.CombinedModel_metrics!=None) & self.combined_model:
            model.add_metrics(self.CombinedModel_metrics)

        return model
    

class RPNloss():
    """Loss helper for training RPN that combines IoU and bbox measures.

    This class computes a loss for region proposal networks. It can also
    optionally compute target confidences from anchor IoU to targets and
    return matching results for downstream heads.
    """
    def __init__(self, 
                 confidence_score=True,
                 anchor_args=None,
                 normalize_target_scores=False,
                 iou_weight=0.5,
                 init_top_k_proposals=0.5, 
                 iou_threshold=0.5, 
                 output_proposals=200,
                 return_matching=True,
                 label_proposals=30,
                 matching_iou_weight=0.5,
                 confidence_loss = tf.keras.losses.MeanAbsoluteError(),
                 name='RPNL',
                 **kwargs):
        """Initialize RPN loss configuration.

        Args:
            confidence_score: whether to predict scores per anchor.
            anchor_args: dict used to generate anchor locations and sizes.
            normalize_target_scores: normalize target compatibility scores.
            iou_weight: weight for IoU component when computing scores.
            init_top_k_proposals/iou_threshold/output_proposals: NMS params.
            return_matching: whether to return matched y_true/y_pred for heads.
            label_proposals: number of proposals used for labeling/matching.
            matching_iou_weight: IoU weight used by matching loss.
            confidence_loss: loss object used for score regression when enabled.
            name: optional name for the loss helper.
        """
        self.name = name
        self.NMS = NonMaxSupression(init_top_k_proposals, iou_threshold, output_proposals)
        self.return_matching = return_matching
        self.matching = MultivariantHungarianLoss(mask=False, losses_weights=[0.,1.,0.], output_proposals=label_proposals, 
                                                  iou_weight=matching_iou_weight, mask_class_pred=True, unit_class_matrix=True)
        
        self.normalize_target_scores = normalize_target_scores
        self.iou_weight = iou_weight
        self.confidence_score = confidence_score

        if confidence_score:
            self.height, self.width = anchor_args['base_img_size']
            bbox_normalization = tf.constant(anchor_args['base_img_size']*2, dtype=tf.float32)[tf.newaxis,tf.newaxis]
            anchors = tf.concat([gen_anchors((None, size, size), anchor_size, window_size, anchor_args['anchor_scales'], anchor_args['base_img_size']) 
                                      for size, anchor_size, window_size in zip(*list(map(anchor_args.get, ['input_sizes', 'anchor_sizes', 'window_sizes'])))], axis=1)
            self.anchors = self._bbox_decoding(anchors)/bbox_normalization
            self.BC = confidence_loss #tf.keras.losses.BinaryCrossentropy(reduction='none')
            self.BC.reduction = tf.keras.losses.Reduction.NONE

    def get_config(self):
        """Return serializable configuration for the loss helper."""
        return {
            'name': self.name,
            'init_top_k_proposals': self.NMS.init_top_k_ratio,
            'iou_threshold': self.NMS.iou_threshold,
            'output_proposals': self.NMS.output_proposals,
            'return_matching': self.return_matching
        }
    
    def _bbox_decoding(self, anchor_bboxes):
        """Decode anchors to absolute XYXY coordinates (helper used internally).

        Args:
            anchor_bboxes: anchors as center+size values.

        Returns:
            Anchors decoded to XYXY clipped to image bounds.
        """
        YX, HW = anchor_bboxes[...,:2], anchor_bboxes[...,2:]
        HW *= 0.5

        return tf.concat([tf.clip_by_value(YX-HW, [0,0], [self.height, self.width]), tf.clip_by_value(YX+HW, [0,0], [self.height, self.width])], axis=-1)
    
    @staticmethod
    def _binary_crossentropy(a,b):
        """Numerically stable binary cross-entropy helper.

        Computes a*(log b) + (1-a)*(log(1-b)) with clipping to avoid log(0).
        """
        b = tf.clip_by_value(b, 1e-5, 1-1e-5)
        return a*tf.math.log(b) + (1-a)*tf.math.log(1-b)
    
    def Ln(self, a, b):
        """L1 distance helper across the last axis.

        Args:
            a, b: tensors to compare.

        Returns:
            Mean absolute difference along the last axis.
        """
        return tf.reduce_mean(tf.abs((a-b)), axis=-1)

    def __call__(self, y_true, y_pred):
        """Compute the RPN loss and optionally return matched targets.

        Args:
            y_true: dict with keys 'bbox' and 'class' describing targets.
            y_pred: dict with keys 'class' and 'bbox' describing predictions.

        Returns:
            If return_matching True, returns (loss_value, y_true_match, y_pred_match),
            otherwise returns loss_value tensor.
        """
        target_bboxes, target_class = y_true['bbox'], y_true['class'] # [B,T,4], [B,T]
        confidence = y_pred['class'] # [B,P], [B,P,4]
        bboxes = y_pred['bbox'] if not self.confidence_score else self.anchors

        T, P = tf.shape(target_bboxes)[1], tf.shape(bboxes)[1]
        target_bboxes = tf.repeat(tf.expand_dims(target_bboxes, axis=1), P, axis=1) # [B,P,T,4]
        pred_bboxes = tf.repeat(tf.expand_dims(bboxes, 2), T, axis=2) # [B,P,T,4]

        if self.confidence_score:
            B = tf.shape(target_bboxes)[0]
            pred_bboxes = tf.repeat(pred_bboxes, B, axis=0)

        scores = self.iou_weight*IoU(target_bboxes, pred_bboxes)+(1-self.iou_weight)*(1-self.Ln(target_bboxes, pred_bboxes)) #[B,P,T]
        if self.normalize_target_scores:
            max_scores = tf.reduce_max(scores, axis=1, keepdims=True) + 1e-6 # [B,1,T]
            scores /= max_scores

        mask = tf.expand_dims(target_class, axis=1) # [B,1,T]
        scores *= mask
        scores = tf.reduce_max(scores, axis=2)+1e-3 # [B,P]

        if not self.confidence_score:
            weighted_confidence = confidence/tf.reduce_sum(confidence, axis=-1, keepdims=True)#tf.nn.softmax(confidence, axis=-1) # [B,P]
            loss_value = 1-tf.reduce_sum(scores*weighted_confidence, axis=-1)
        else:
            confidence = tf.clip_by_value(confidence, 1e-5, 1-1e-5)
            loss_value = self.BC(scores, confidence)

        if self.return_matching:
            if self.confidence_score:
                bboxes = tf.repeat(bboxes, B, axis=0)
            confidence, bboxes = self.NMS([confidence, bboxes])
            _, y_true_match, y_pred_match = self.matching(y_true,{'class': confidence, 'bbox': bboxes})
            return loss_value, y_true_match, y_pred_match
        return loss_value