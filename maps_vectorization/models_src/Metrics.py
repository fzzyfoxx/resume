import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from models_src.VecModels import flatten

def norm(x, reg=1.):
    """Standardize a tensor by subtracting its mean and dividing by std * reg.

    Args:
        x: Input tensor.
        reg: Regularization factor multiplied with the standard deviation
            before division. Helps control scaling and numerical stability.

    Returns:
        Tensor of the same shape as `x` with normalized values.
    """
    return (x-tf.reduce_mean(x))/(tf.math.reduce_std(x)*reg+1e-4)

def norm_weights(w, batch_dims=0):
    """Normalize weights by the number of elements along non-batch dimensions.

    Args:
        w: Weight tensor.
        batch_dims: Number of leading (batch) dimensions that should be
            excluded from the normalization factor.

    Returns:
        Tensor of normalized weights with the same shape as `w`.
    """
    return w/tf.cast(tf.reduce_prod(tf.shape(w)[batch_dims:]), w.dtype)

def weighted_sum(x, weights, axis=None, keepdims=False):
    """Compute a weighted sum of `x` along a given axis.

    Args:
        x: Input tensor.
        weights: Tensor of weights broadcastable to `x`.
        axis: Axis or axes along which to reduce. If `None`, reduces over all
            dimensions.
        keepdims: If True, retains reduced dimensions with length 1.

    Returns:
        Tensor containing the weighted sum.
    """
    return tf.reduce_sum(x*weights, axis=axis, keepdims=keepdims)

def weighted_std(x, weights, x_mean, axis=None, keepdims=None):
    """Compute weighted standard deviation of a tensor.

    Args:
        x: Input tensor.
        weights: Tensor of weights broadcastable to `x`.
        x_mean: Precomputed (weighted) mean of `x` with shape broadcastable
            to `x`.
        axis: Axis or axes along which to reduce. If `None`, reduces over all
            dimensions.
        keepdims: If True, retains reduced dimensions with length 1.

    Returns:
        Tensor of weighted standard deviation along the specified axes.
    """
    return tf.reduce_sum(input_tensor=(x-x_mean)**2*weights, axis=axis, keepdims=keepdims)**0.5

def adaptive_loss_weights(weights, loss_values, reg=1.):
    """Adapt weights based on per-sample loss values.

    This function rescales the provided `weights` so that samples with higher
    normalized loss receive higher weights, while preserving the original
    weight structure.

    Args:
        weights: Tensor of sample weights.
        loss_values: Tensor of per-sample loss values broadcastable to
            `weights`.
        reg: Regularization factor applied to the normalized loss to control
            sharpness of the weighting.

    Returns:
        Tensor of adapted weights with the same shape as `weights`.
    """

    flat_weights = flatten(weights)
    #s = tf.reduce_sum(flat_weights)
    flat_loss_values = flatten(loss_values)
    norm_w = flat_weights/(tf.reduce_sum(flat_weights, axis=-1, keepdims=True)+1e-6) #norm_weights(flat_weights, batch_dims=1)
    weights_sum = tf.reduce_sum(norm_w, axis=-1, keepdims=True)
    x_mean = weighted_sum(flat_loss_values, norm_w, axis=-1, keepdims=True)
    x_std = weighted_std(flat_loss_values, norm_w, x_mean, axis=-1, keepdims=True)

    loss_norm = (flat_loss_values*weights_sum-x_mean)/(x_std*reg+1e-6)

    sm_nom = tf.math.exp(loss_norm)*weights_sum

    sm_denom = tf.reduce_sum(sm_nom*norm_w, axis=-1, keepdims=True)

    adapted_weights = tf.reshape(sm_nom/(sm_denom+1e-6)*flat_weights, tf.shape(weights))

    return adapted_weights

class AdaptiveWeightsLoss(tf.keras.Loss):
    """Wrapper loss that adaptively reweights samples based on their losses.

    The base `loss_func` is evaluated and the provided `sample_weight` is
    adjusted on-the-fly so that harder examples receive higher weights.
    """

    def __init__(self, loss_func, reg=20., adapt_ratio=0.5, norm_clip=2., reduction='sum_over_batch_size', **kwargs):
        """Initialize the adaptive loss wrapper.

        Args:
            loss_func: A `tf.keras.losses.Loss` instance to be wrapped.
            reg: Regularization factor applied to the standard deviation when
                normalizing losses.
            adapt_ratio: Interpolation factor between original and adapted
                sample weights (0 = no adaptation, 1 = full adaptation).
            norm_clip: Absolute value used to clip the normalized losses.
            reduction: Loss reduction method passed to the base `Loss`.
            **kwargs: Additional keyword arguments passed to `tf.keras.Loss`.
        """
        super().__init__(reduction=reduction,**kwargs)

        self.loss_func = loss_func
        self.reg = reg
        self.adapt_ratio = adapt_ratio
        self.norm_clip = norm_clip

    def call(self, y_true, y_pred):
        """Compute the underlying loss without any weighting adaptation.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.

        Returns:
            Tensor of loss values as returned by `self.loss_func`.
        """
        return self.loss_func.call(y_true, y_pred)
    
    def adaptive_loss_weights(self, weights, loss_values):
        """Compute adapted sample weights from current losses.

        Args:
            weights: Original sample weight tensor.
            loss_values: Per-sample loss tensor.

        Returns:
            Tensor of adapted sample weights with the same shape as `weights`.
        """

        flat_weights = flatten(weights)
        #s = tf.reduce_sum(flat_weights)
        flat_loss_values = flatten(loss_values)
        norm_w = flat_weights/(tf.reduce_sum(flat_weights, axis=-1, keepdims=True)+1e-6) #norm_weights(flat_weights, batch_dims=1)
        weights_sum = tf.reduce_sum(norm_w, axis=-1, keepdims=True)
        x_mean = weighted_sum(flat_loss_values, norm_w, axis=-1, keepdims=True)
        x_std = weighted_std(flat_loss_values, norm_w, x_mean, axis=-1, keepdims=True)

        loss_norm = tf.clip_by_value((flat_loss_values*weights_sum-x_mean)/(x_std*self.reg+1e-6), -self.norm_clip, self.norm_clip)

        sm_nom = tf.math.exp(loss_norm)*weights_sum

        sm_denom = tf.reduce_sum(sm_nom*norm_w, axis=-1, keepdims=True)

        adapted_weights = tf.reshape(sm_nom/(sm_denom+1e-6)*flat_weights, tf.shape(weights))

        return adapted_weights
        
    def adapt_weights(self, y_true, y_pred, sample_weight):
        """Blend original and adapted sample weights.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.
            sample_weight: Original sample weights.

        Returns:
            Tensor of blended sample weights.
        """
        losses = self.loss_func.call(y_true, y_pred)
        adapted_weight = self.adaptive_loss_weights(sample_weight, losses)

        return (1-self.adapt_ratio)*sample_weight + self.adapt_ratio*adapted_weight
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Compute the adapted loss.

        If `sample_weight` is provided, it is updated based on the current
        loss values in a gradient-stopped fashion and passed to the parent
        `Loss.__call__`.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.
            sample_weight: Optional tensor of initial sample weights.

        Returns:
            Scalar loss tensor reduced according to `self.reduction`.
        """

        if sample_weight is not None:
            sample_weight = tf.stop_gradient(self.adapt_weights(y_true, y_pred, sample_weight))
        return super().__call__(y_true, y_pred, sample_weight)

class LossBasedMetric(tf.keras.metrics.Mean):
    """Metric that reports the mean of a given loss function.

    The wrapped loss is forced to have `NONE` reduction and its average over
    all elements is tracked as a scalar metric.
    """

    def __init__(self, loss_func, **kwargs):
        """Initialize the metric.

        Args:
            loss_func: A `tf.keras.losses.Loss` instance to evaluate.
            **kwargs: Additional keyword arguments passed to `tf.keras.metrics.Mean`.
        """
        super().__init__(**kwargs)

        self.loss_func = loss_func
        self.loss_func.reduction = tf.keras.losses.Reduction.NONE

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the running mean with a new batch.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.
            sample_weight: Optional sample weights passed to the loss.

        Returns:
            Update op from the parent `Mean` metric.
        """
        values = self.loss_func(y_true, y_pred, sample_weight)
        values = tf.reduce_sum(values)/tf.cast(tf.reduce_prod(tf.shape(values)), tf.float32)
        return super().update_state(values, None)

class F12D(tf.keras.metrics.Metric):
    """F1-score metric wrapper that works on 2D outputs.

    Internally uses `tf.keras.metrics.F1Score` with `micro` averaging and
    flattens spatial dimensions.
    """

    def __init__(self, threshold, name='F1', **kwargs):
        """Initialize the F1 metric.

        Args:
            threshold: Decision threshold applied to predictions.
            name: Metric name.
            **kwargs: Additional keyword arguments passed to `tf.keras.metrics.Metric`.
        """
        super(F12D, self).__init__(name=name, **kwargs)

        self.f1 = tf.keras.metrics.F1Score(threshold=threshold, average='micro')
        self.flatten = tf.keras.layers.Flatten()
        self.score = self.add_weight(name='f1', initializer='zeros')
        self.iterations = self.add_weight(name='iters', initializer='zeros')
        self.threshold = threshold

    def get_config(self):
        """Return the metric configuration for serialization.

        Returns:
            A Python dict with metric configuration, including `threshold`.
        """
        return {**super().get_config(), 'threshold': self.threshold}

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update F1-score state with a new batch.

        Args:
            y_true: Ground-truth labels.
            y_pred: Model predictions.
            sample_weight: Optional sample weights (ignored here).
        """

        self.score.assign_add(self.f1(self.flatten(tf.cast(y_true, tf.float32)), self.flatten(y_pred)))
        self.iterations.assign_add(1.0)

    def result(self):
        """Compute the average F1-score over all updates.

        Returns:
            Scalar tensor containing the mean F1-score.
        """
        return self.score/self.iterations
    
class WeightedF12D(tf.keras.metrics.Metric):
    """F1-style metric computed from Precision and Recall metrics.

    This metric keeps separate `Precision` and `Recall` instances (with a
    fixed threshold) and combines them into an F1-score.
    """

    def __init__(self, name='F1', threshold=0.5, average='micro', **kwargs):
        """Initialize the weighted F1 metric.

        Args:
            name: Metric name.
            threshold: Decision threshold for precision/recall.
            average: Unused, kept for API compatibility.
            **kwargs: Additional keyword arguments passed to `tf.keras.metrics.Metric`.
        """
        super().__init__(name=name, **kwargs)

        self.score = tf.keras.metrics.Mean()
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the internal F1-score estimate.

        Args:
            y_true: Ground-truth labels.
            y_pred: Model predictions.
            sample_weight: Optional sample weights for precision/recall.
        """
        prec = self.precision(y_true, y_pred, sample_weight)
        rec = self.recall(y_true, y_pred, sample_weight)

        score = tf.math.divide_no_nan(2*prec*rec, prec+rec)

        self.score.update_state(score)

    def reset_state(self):
        """Reset all internal state variables."""
        self.score.reset_state()
        self.precision.reset_state()
        self.recall.reset_state()

    def result(self):
        """Return the current F1-score estimate.

        Returns:
            Scalar tensor with the running mean F1-score.
        """
        return self.score.result()

def extract_coords(bbox):
    """Split bounding box tensor into coordinate components.

    Assumes the last dimension encodes `[y1, x1, y2, x2]`.

    Args:
        bbox: Tensor of bounding boxes.

    Returns:
        Tuple `(y1, x1, y2, x2)` of tensors with the same leading shape
        as `bbox` and last dimension removed.
    """
    return bbox[...,0], bbox[...,1], bbox[...,2], bbox[...,3]
    
def calc_area(Y1,X1,Y2,X2):
    """Compute the area of axis-aligned bounding boxes.

    Args:
        Y1: Top y-coordinate tensor.
        X1: Left x-coordinate tensor.
        Y2: Bottom y-coordinate tensor.
        X2: Right x-coordinate tensor.

    Returns:
        Tensor with box areas.
    """
    return tf.abs(X2-X1)*tf.abs(Y2-Y1)

def IoU(a,b):
    """Compute the Intersection-over-Union (IoU) between two box sets.

    Args:
        a: Tensor of bounding boxes with last dimension `[y1, x1, y2, x2]`.
        b: Tensor of bounding boxes with last dimension `[y1, x1, y2, x2]`.

    Returns:
        Tensor of IoU values broadcast over leading dimensions of `a` and `b`.
    """
    aY1, aX1, aY2, aX2 = extract_coords(a)
    bY1, bX1, bY2, bX2 = extract_coords(b)

    xI = tf.abs(tf.clip_by_value(bX2, aX1, aX2) - tf.clip_by_value(bX1, aX1, aX2))
    yI = tf.abs(tf.clip_by_value(bY2, aY1, aY2) - tf.clip_by_value(bY1, aY1, aY2))

    I = xI*yI
    U = calc_area(aY1, aX1, aY2, aX2)+calc_area(bY1, bX1, bY2, bX2)-I

    return tf.math.divide_no_nan(I,U)

class MultiLevelFocalCrossentropy(tf.keras.losses.Loss):
    """Binary focal cross-entropy applied at multiple pooled resolutions.

    The loss is computed on the original resolution and on several levels of
    max-pooled versions of the inputs, and then averaged.
    """

    def __init__(self, pooling_levels=2, alpha=0.75, gamma=2, reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        """Initialize the multi-level focal loss.

        Args:
            pooling_levels: Number of max-pooling levels applied on top of the
                base resolution.
            alpha: Class balancing factor for focal loss.
            gamma: Focusing parameter for focal loss.
            reduction: Loss reduction strategy.
            **kwargs: Additional keyword arguments passed to `tf.keras.losses.Loss`.
        """
        super().__init__(name="MLBFC",reduction=reduction,**kwargs)
        
        self.alpha = alpha
        self.gamma = gamma
        self.pooling_levels = pooling_levels
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true, y_pred):
        """Compute the multi-level focal loss.

        Args:
            y_true: Ground-truth binary labels.
            y_pred: Predicted probabilities.

        Returns:
            Tensor of loss values, averaged across pooling levels.
        """
        loss = tf.keras.losses.binary_focal_crossentropy(self.flatten(y_true), self.flatten(y_pred), apply_class_balancing=True, alpha=self.alpha, gamma=self.gamma)
        for i in range(self.pooling_levels):
            y_true = tf.nn.max_pool2d(y_true, 3, 2, 'VALID')
            y_pred = tf.nn.max_pool2d(y_pred, 3, 2, 'VALID')
            loss += tf.keras.losses.binary_focal_crossentropy(self.flatten(y_true), self.flatten(y_pred), apply_class_balancing=True, alpha=self.alpha, gamma=self.gamma)

        return loss/(1+self.pooling_levels)
    

class IoUMetric(tf.keras.metrics.Metric):
    """Mean Intersection-over-Union metric for bounding boxes."""

    def __init__(self, name='IoU'):
        """Initialize the IoU metric.

        Args:
            name: Metric name.
        """
        super().__init__(name=name)

        self.score = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state with new predictions.

        Args:
            y_true: Ground-truth bounding boxes.
            y_pred: Predicted bounding boxes.
            sample_weight: Optional weights for each box.
        """

        scores = tf.expand_dims(IoU(y_true, y_pred), axis=-1)

        self.score.update_state(scores, sample_weight=sample_weight)

    def reset_state(self):
        """Reset the internal running mean."""
        self.score.reset_state()

    def result(self):
        """Return the current mean IoU.

        Returns:
            Scalar tensor with the running mean IoU.
        """
        return self.score.result()

class HungarianMatching:
    """Performs Hungarian matching based on a given cost function.

    This utility pools inputs, reshapes them into proposals and targets, and
    then uses the Hungarian algorithm to find an optimal assignment.
    """

    def __init__(self, matching_loss_func=tf.keras.losses.BinaryCrossentropy(), pool_size=4, perm=[0,2,1]):
        """Initialize the Hungarian matcher.

        Args:
            matching_loss_func: Loss function used to build the cost matrix.
            pool_size: Optional spatial pooling factor before matching.
            perm: Permutation applied to reshaped tensors before loss.
        """

        self.matching_loss = matching_loss_func
        self.matching_loss.reduction = tf.keras.losses.Reduction.NONE

        self.pool_size = pool_size
        self.perm = perm

    def _calc_losses(self, cost_matrix):
        """Extract matched losses from a cost matrix.

        Args:
            cost_matrix: Tensor of pairwise costs.

        Returns:
            Tensor containing the cost associated with each matched pair.
        """
        match_idxs = tf.map_fn(lambda x: tf.transpose(tf.convert_to_tensor(linear_sum_assignment(x)), perm=[1,0]), elems=cost_matrix, fn_output_signature=tf.int32)
        return tf.gather_nd(cost_matrix, match_idxs, batch_dims=1)
    
    def _gen_matching(self, cost_matrix, y_true, y_pred):
        """Generate matched `y_true` and `y_pred` according to the cost matrix.

        Args:
            cost_matrix: Tensor of pairwise costs.
            y_true: Ground-truth tensor.
            y_pred: Prediction tensor.

        Returns:
            Tuple `(matched_y_true, matched_y_pred)` with reordered entries.
        """
        match_idxs = tf.map_fn(lambda x: tf.convert_to_tensor(linear_sum_assignment(x)), elems=cost_matrix, fn_output_signature=tf.int32)
        return tf.gather(tf.squeeze(y_true, axis=1), match_idxs[:,1], batch_dims=1, axis=1), \
            tf.gather(tf.squeeze(y_pred, axis=2), match_idxs[:,0], batch_dims=1, axis=1)

    @tf.function
    def __call__(self, y_true, y_pred):
        """Compute Hungarian matching indices for a batch.

        Args:
            y_true: Ground-truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Tensor of shape `[B, N, 2]` with (pred_idx, true_idx) pairs
            representing the optimal assignment for each batch element.
        """
        if self.pool_size>1:
            y_true_pooled = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=self.pool_size)(y_true)
            y_pred_pooled = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=self.pool_size)(y_pred)
        else:
            y_true_pooled = y_true
            y_pred_pooled = y_pred

        pooled_shape = tf.shape(y_true_pooled)
        B, T = pooled_shape[0], pooled_shape[-1]
        P = tf.shape(y_pred_pooled)[-1]

        y_true_pooled = tf.expand_dims(tf.transpose(tf.reshape(y_true_pooled, (B,-1,T)), perm=self.perm), axis=1)
        y_pred_pooled = tf.expand_dims(tf.transpose(tf.reshape(y_pred_pooled, (B,-1,P)), perm=self.perm), axis=2)

        cost_matrix = self.matching_loss(y_true_pooled, y_pred_pooled)

        match_idxs = tf.map_fn(lambda x: tf.transpose(tf.convert_to_tensor(tf.numpy_function(linear_sum_assignment, [x], [tf.int64, tf.int64])), perm=[1,0]), elems=cost_matrix, fn_output_signature=tf.int64)

        return match_idxs

class HungarianLoss(tf.keras.losses.Loss):
    """Loss based on Hungarian matching between predictions and targets."""

    def __init__(self, hungarian_matching, loss_func=tf.keras.losses.BinaryCrossentropy(), reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        """Initialize the Hungarian loss.

        Args:
            hungarian_matching: A `HungarianMatching` instance.
            loss_func: Base loss applied after reordering predictions/targets.
            reduction: Loss reduction strategy.
            **kwargs: Additional keyword arguments passed to `tf.keras.losses.Loss`.
        """
        super().__init__(name="HL",reduction=reduction,**kwargs)
        
        self.hungarian_matching = hungarian_matching
        self.loss_func = loss_func

    def call(self, y_true, y_pred):
        """Apply Hungarian matching then evaluate the base loss.

        Args:
            y_true: Ground-truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Loss tensor as returned by `self.loss_func` after matching.
        """
        y_true = tf.cast(y_true, tf.float32)
        match_idxs = tf.stop_gradient(self.hungarian_matching(y_true, y_pred))
        y_true = tf.gather(y_true, match_idxs[:,:,1], batch_dims=1, axis=-1)
        y_pred = tf.gather(y_pred, match_idxs[:,:,0], batch_dims=1, axis=-1)
        return self.loss_func(y_true, y_pred)
    
class HungarianClassificationMetric(tf.keras.metrics.Metric):
    """Metric that evaluates a classification metric after Hungarian matching."""

    def __init__(self, hungarian_matching, metric_func, name='ClassMetric', **kwargs):
        """Initialize the classification metric.

        Args:
            hungarian_matching: A `HungarianMatching`-compatible callable.
            metric_func: Callable taking `(y_true, y_pred)` and returning
                a scalar metric value.
            name: Metric name.
            **kwargs: Additional keyword arguments passed to `tf.keras.metrics.Metric`.
        """
        super(HungarianClassificationMetric, self).__init__(name=name, **kwargs)

        self.hungarian_matching = hungarian_matching
        self.metric_func = metric_func
        #self.flatten = tf.keras.layers.Flatten()

        self.score = self.add_weight(name=name+'_score', initializer='zeros') #tf.Variable(0, dtype=tf.float32)#
        self.iterations = self.add_weight(name=name+'_iters', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state using Hungarian-matched pairs.

        Args:
            y_true: Ground-truth tensor.
            y_pred: Predicted tensor.
            sample_weight: Unused; present for API compatibility.
        """
        y_true = tf.cast(y_true, tf.float32)
        shape = tf.shape(y_true)
        B, H, W, T = shape[0], shape[1], shape[2], shape[3]

        match_idxs = tf.stop_gradient(self.hungarian_matching(y_true, y_pred))
        y_true = tf.reshape(tf.gather(y_true, match_idxs[:,:,1], batch_dims=1, axis=-1), (B, H*W*T))
        y_pred = tf.reshape(tf.gather(y_pred, match_idxs[:,:,0], batch_dims=1, axis=-1), (B, H*W*T))
        self.score.assign_add(self.metric_func(y_true, y_pred))
        self.iterations.assign_add(1.0)

    def reset_states(self):
        """Reset the internal accumulated metric values."""
        self.score.assign(0.0)
        self.iterations.assign(0.0)

    def result(self):
        """Return the mean metric value over all updates.

        Returns:
            Scalar tensor with the accumulated mean metric value.
        """
        return self.score/self.iterations
    

class MultivariantHungarianLoss():
    """Composite Hungarian loss for detection-style multi-output models.

    This class builds a *joint* Hungarian cost matrix from several modalities
    (classification scores, bounding boxes, and masks), runs Hungarian
    matching once per image, and then aggregates losses for the matched
    pairs.

    The workflow is:

    1. For each enabled component (class / bbox / mask):
       * Optionally pool the tensors spatially.
       * Permute and reshape predictions and targets into
         `[batch, proposals, features]` and `[batch, targets, features]`.
       * Tile them so a pairwise cost is computed for *every* proposal–target
         combination using the provided loss function.
       * Optionally replace the true class tensor with an all-ones tensor
         (`unit_class_matrix=True`) to build a class-agnostic cost.
    2. Normalize and weight the component cost matrices with
       `losses_weights`, and sum them into a single cost matrix.
    3. Run the Hungarian algorithm on this joint cost matrix to obtain a
       one-to-one assignment between proposals and targets per batch item.
    4. Optionally apply object-presence masks (`mask_class_pred` /
       `masking_flags`) when aggregating losses so that background slots do
       not dominate the objective.
    5. Optionally return the matched predictions and targets for each
       component (useful for debugging, logging, or auxiliary heads).

    This design makes it easy to:
    * Turn individual components on/off (`classification`, `bbox`, `mask`).
    * Re-weight components via `losses_weights` without changing code.
    * Trade off IoU vs. coordinate regression (`iou_weight`,
      `bbox_regularization_rank`).
    * Control focal loss behavior on masks (`mask_alpha`, `mask_gamma`,
      `mask_smoothing`, `mask_pool_size`).
    * Use a class-presence mask to restrict which proposals contribute to
      the mask and bbox components (`mask_class_pred`).
    * Either use the class labels themselves or an all-ones matrix to build
      the class term in the cost (`unit_class_matrix`).
    """

    def __init__(self,
                 classification=True,
                 bbox=True,
                 mask=True,
                 losses_weights=[1,1,1],
                 output_proposals=6,
                 output_mask_size=(256,256),
                 iou_weight=0.5,
                 bbox_regularization_rank=1,
                 mask_alpha=0.75,
                 mask_gamma=5,
                 mask_pool_size=8,
                 mask_class_pred=False,
                 return_matching=True,
                 name='HL',
                 class_smoothing=0.0,
                 mask_smoothing=0.0,
                 unit_class_matrix=False,
                 **kwargs):
        """Configure a multi-component Hungarian loss.

        Args:
            classification: If True, include a classification term in both the
                Hungarian cost matrix and the final loss. Disabling this
                removes classification from matching and loss computation.
            bbox: If True, include a bounding-box term in the cost and final
                loss. The term is a combination of IoU and coordinate
                regression (`IoULoss` and `Ln`).
            mask: If True, include a mask term (binary focal loss) in the
                cost and final loss.
            losses_weights: Relative weights for the enabled components in
                `[classification, bbox, mask]` order. Only the entries
                corresponding to active components are used and then
                normalized to sum to 1 when combining component costs.
            output_proposals: Number of proposal slots per image used when
                reshaping matched outputs (controls the static shape of the
                returned features for each component).
            output_mask_size: Spatial size `(H, W)` used when flattening
                masks to a vector per proposal.
            iou_weight: Weight of the IoU part in the bbox loss. The final
                bbox loss is:

                `iou_weight * IoULoss + (1 - iou_weight) * Ln`.

            bbox_regularization_rank: Power used in coordinate regression in
                `Ln`. For example, 1 gives L1-like behavior, 2 gives an
                L2-like penalty.
            mask_alpha: Alpha parameter of focal loss for masks.
            mask_gamma: Gamma parameter of focal loss for masks.
            mask_pool_size: Spatial pooling factor applied to masks when
                building the mask cost matrix. This trades detail for speed
                and stability.
            mask_class_pred: If True, use the predicted/true object
                classification to mask the bbox and mask components so that
                only object slots contribute.
            return_matching: If True, `__call__` returns both the per-sample
                losses and dictionaries of matched predictions and targets.
                If False, only the loss tensor is returned.
            name: Identifying name for configuration/serialization.
            class_smoothing: Label smoothing factor applied to the
                classification loss when building the cost matrix.
            mask_smoothing: Label smoothing factor applied to the mask loss
                when building the cost matrix.
            unit_class_matrix: If True, when constructing the classification
                term of the cost, the true class tensor is replaced by an
                all-ones matrix. This makes the class component independent
                of the exact labels and effectively enforces only that
                predictions match a generic "object" pattern.
            **kwargs: Extra keyword arguments kept for API compatibility.
        """
        #super(MultivariantHungarianLoss, self).__init__(name='HL', **kwargs)
        self.name = name

        self.iou_weight = iou_weight
        self.bbox_reg_rank = bbox_regularization_rank
        self.unit_class_matrix = unit_class_matrix


        class_loss = tf.keras.losses.BinaryCrossentropy(reduction='none', label_smoothing=class_smoothing)
        mask_loss = tf.keras.losses.BinaryFocalCrossentropy(reduction='none', alpha=mask_alpha, gamma=mask_gamma, label_smoothing=mask_smoothing)
        flags=[classification, bbox, mask]
        self.inputs_num = sum(flags)

        self.cost_matrix_args, self.masking_flags, self.input_names, self.matching_args, self.losses_weights = self._gen_cost_matrix_args(flags=flags,
                                            losses=[class_loss, self.BBoxLoss, mask_loss],
                                            pool_sizes=[1,1,mask_pool_size],
                                            perms=[[0,1],[0,1,2],[0,3,1,2]],
                                            class_masks=[mask_class_pred, True, True],
                                            input_names=['class', 'bbox', 'mask'],
                                            output_sizes=[(output_proposals,1), (output_proposals,4), (output_proposals,output_mask_size[0]*output_mask_size[1])],
                                            losses_weights=losses_weights
        )
        
        self.classification = classification

        self.return_matching = return_matching

        self.pool = tf.keras.layers.AveragePooling2D(pool_size=mask_pool_size, strides=mask_pool_size)

    ##### Cost Matrix #####

    def _gen_cost_matrix_args(self, flags, losses, pool_sizes, perms, class_masks, input_names, output_sizes, losses_weights):
        """Build internal configuration for cost-matrix construction.

        For each potential component (classification, bbox, mask), this
        method:
        * Checks the corresponding `flag`.
        * If enabled, stores how to compute the component cost matrix
          (loss function, pooling and permutation strategy, and whether
          to replace the true labels with a unit matrix).
        * Records whether the component should be masked by the object
          presence mask (`class_masks`).
        * Prepares arguments used later when extracting matched features
          (`matching_args`).

        Args:
            flags: List of booleans indicating which components are enabled.
            losses: List of loss callables for each component.
            pool_sizes: List of pooling sizes per component used when
                building the cost matrices.
            perms: List of permutations applied to align tensors before
                reshaping into `[batch, entities, features]`.
            class_masks: Per-component booleans indicating if that component
                uses the object mask when aggregating component losses.
            input_names: Logical names (e.g. "class", "bbox", "mask") used
                as keys for inputs and outputs.
            output_sizes: Static output sizes for each component, used to set
                shapes of returned matched features.
            losses_weights: Raw weights corresponding to each component
                before normalization.

        Returns:
            Tuple of:
                * cost_matrix_args: List of dictionaries describing how to
                  build each component cost matrix.
                * masking_flags: Boolean list mirroring `class_masks` for the
                  enabled components.
                * names: Names of the enabled components.
                * matching_args: List of dictionaries that describe how to
                  extract matched features for each component.
                * losses_weights_tensor: 1D float tensor of normalized
                  component weights summing to 1.
        """
        output_args = []
        masking_flags = []
        names = []
        matching_args = []
        losses_weights_list = []
        for flag, loss, pool_size, perm, class_mask, name, output_size, loss_weight in zip(flags, losses, pool_sizes, perms, class_masks, input_names, output_sizes, losses_weights):
            if flag:
                output_args.append({
                    'cost_func': loss,
                    'pool_size': pool_size,
                    'perm': perm,
                    'unit_true': name=='class' if self.unit_class_matrix else False
                    })
                masking_flags.append(class_mask)
                names.append(name)

                matching_args.append({
                    'class_mask': class_mask,
                    'perm': perm,
                    'output_size': output_size
                })
                losses_weights_list.append(loss_weight)
        
        losses_weights_list = tf.constant(losses_weights_list, dtype=tf.float32)
        losses_weights_list /= tf.reduce_sum(losses_weights_list)

        return output_args, masking_flags, names, matching_args, losses_weights_list
    
    def _calc_cost_matrix(self, a, b, cost_func,pool_size=1, perm=[0,1,2], unit_true=False):
        """Compute a pairwise cost matrix for a single component.

        Given a set of ground-truth items `a` and predictions `b`, this
        method:
        * Optionally pools both tensors.
        * Permutes and reshapes them into
          `[batch, targets, features]` and `[batch, proposals, features]`.
        * Tiles both so each proposal is compared to every target.
        * Calls `cost_func` to obtain pairwise costs.

        Args:
            a: Ground-truth tensor for a component.
            b: Prediction tensor for a component.
            cost_func: Loss function that produces pairwise costs over the
                last dimension.
            pool_size: Spatial pooling factor before cost computation.
            perm: Permutation applied before reshaping.
            unit_true: If True, replace `a` with an all-ones tensor before
                computing the cost (used when `unit_class_matrix=True`).

        Returns:
            Tensor of shape `[batch, proposals, targets]` (or analogous)
            containing pairwise costs between proposals and targets.
        """

        if pool_size>1:
            a = self.pool(a)
            b = self.pool(b)
            
        a = tf.transpose(a, perm=perm)
        b = tf.transpose(b, perm=perm)
        a_shape = tf.shape(a)
        B, T = a_shape[0], a_shape[1]
        P = tf.shape(b)[1]

        if unit_true:
            a = tf.ones_like(a, dtype=tf.float32)

        a = tf.repeat(tf.expand_dims(tf.reshape(a, (B,T,-1)), axis=1), P, axis=1)
        b = tf.repeat(tf.expand_dims(tf.reshape(b, (B,P,-1)), axis=2), T, axis=2)
        cost_matrix = cost_func(a, b)
        
        return cost_matrix


    ##### BBox Loss #####

    ### IoU
    def IoULoss(self, a, b):
        """IoU-based component of the bounding-box loss.

        This is defined as `1 - IoU(a, b)` with clipping to avoid numerical
        issues at 0 and 1.

        Args:
            a: Ground-truth bounding boxes.
            b: Predicted bounding boxes.

        Returns:
            Tensor of IoU-based loss values with the same leading shape as
            the input box tensors.
        """
        return 1-tf.clip_by_value(IoU(a,b), 1e-5, 1-1e-5)
    
    ### Bbox regression
    def Ln(self, a, b):
        """Coordinate regression term for bounding boxes.

        The regression is applied element-wise on box coordinates and then
        averaged across the last dimension. The exponent
        `bbox_regularization_rank` controls how strongly large errors are
        penalized.

        Args:
            a: Ground-truth bounding boxes.
            b: Predicted bounding boxes.

        Returns:
            Tensor of per-box regression losses.
        """
        return tf.reduce_mean(tf.abs((a-b)**self.bbox_reg_rank), axis=-1)
    
    ### Combined BBox Loss
    def BBoxLoss(self, a, b):
        """Combined IoU and regression loss for bounding boxes.

        The final bbox loss is a convex combination of `IoULoss` and `Ln`:

        `iou_weight * IoULoss(a, b) + (1 - iou_weight) * Ln(a, b)`.

        Args:
            a: Ground-truth bounding boxes.
            b: Predicted bounding boxes.

        Returns:
            Tensor of combined bbox loss values.
        """
        return self.iou_weight*self.IoULoss(a,b) + (1-self.iou_weight)*self.Ln(a,b)
    

    ##### Hungarian Matching #####

    @tf.autograph.experimental.do_not_convert
    def HungarianMatching(self, cost_matrix):
        """Run Hungarian algorithm on the joint cost matrix.

        The input cost matrix is assumed to encode the full proposal–target
        costs after combining all enabled components. The Hungarian algorithm
        produces an optimal one-to-one assignment between proposals and
        targets for each batch element.

        Args:
            cost_matrix: Tensor of pairwise costs of shape
                `[batch, proposals, targets]`.

        Returns:
            Tuple `(y_true_idxs, y_pred_idxs)` with integer index tensors
            describing the optimal matching. Indexing with these tensors
            reorders targets and predictions into matched order.
        """
        match_idxs = tf.map_fn(lambda x: tf.transpose(tf.convert_to_tensor(tf.numpy_function(linear_sum_assignment, [x], [tf.int64, tf.int64])), perm=[1,0]), elems=cost_matrix, fn_output_signature=tf.int64)
        y_true_idxs = match_idxs[...,1]
        y_pred_idxs = match_idxs[...,0]
        return y_true_idxs, y_pred_idxs
    
    def add(self, cost_matrices):
        """Combine multiple component cost matrices or loss vectors.

        This method applies the normalized `losses_weights` to each
        component and sums the results. It is used both when constructing
        the joint Hungarian cost matrix and when aggregating per-component
        matched losses.

        Args:
            cost_matrices: List of tensors with identical shapes, one per
                active component.

        Returns:
            Tensor representing the weighted sum of all components.
        """
        cost_matrix = cost_matrices[0] * self.losses_weights[0]
        for i,cm in enumerate(cost_matrices[1:]):
            cost_matrix += cm * self.losses_weights[i+1]
        return cost_matrix
    
    @staticmethod
    def extract_losses(cost_matrix, y_true_idxs, y_pred_idxs):
        """Gather losses corresponding to matched proposal–target pairs.

        Args:
            cost_matrix: Tensor of pairwise costs of shape
                `[batch, proposals, targets]`.
            y_true_idxs: Tensor of matched target indices.
            y_pred_idxs: Tensor of matched proposal indices.

        Returns:
            Tensor of shape `[batch, num_matches]` containing the losses
            associated with each matched pair.
        """
        return tf.gather(tf.gather(cost_matrix, y_pred_idxs, batch_dims=1, axis=1), y_true_idxs, batch_dims=2, axis=2)
    

    def extract_features(self, features, idxs, mask, class_mask, perm, output_size):
        """Extract and optionally mask matched features.

        This helper reshapes and permutes feature maps so that the indices
        coming from Hungarian matching (`idxs`) can be used to gather the
        per-proposal features in matched order. Optionally, an object mask
        is applied to zero-out background entries.

        Args:
            features: Input feature tensor (e.g., class scores or masks).
            idxs: Integer index tensor specifying which entries to gather
                along the proposal dimension.
            mask: Optional float tensor used when `class_mask` is True to
                mask out background proposals.
            class_mask: If True, multiply extracted features by `mask`.
            perm: Permutation applied before reshaping and gathering.
            output_size: Static output shape (excluding batch) used to set
                the final tensor shape for downstream layers.

        Returns:
            Tensor of matched features with shape `(batch, *output_size)`.
        """
        idxs_shape = tf.shape(idxs)
        B, T = idxs_shape[0], idxs_shape[1]
        features = tf.transpose(features, perm=perm)
        P = tf.shape(features)[1]
        features = tf.reshape(features, (B,P,-1))
        
        features = tf.gather(features, idxs, batch_dims=1, axis=1)

        if class_mask:
            features *= mask
        
        #features = tf.reshape(features, (B,-1))
        features.set_shape((None,)+output_size)
        return features

    ##########

    def get_config(self):
        """Return a serializable configuration of this loss.

        The returned dictionary contains the most important constructor
        arguments and flags so that an equivalent `MultivariantHungarianLoss`
        instance can be recreated.

        Returns:
            A Python dict with configuration keys such as `name`,
            `input_names`, `losses_weights`, `iou_weight`, and others.
        """
        return {
            'name': self.name,
            'input_names': self.input_names,
            'losses_weights': self.losses_weights,
            'iou_weight': self.iou_weight,
            'bbox_reg_rank': self.bbox_reg_rank,
            'unit_class_matrix': self.unit_class_matrix,
            'masking_flags': self.masking_flags,
            'return_matching': self.return_matching
        }
    
    def __call__(self, y_true, y_pred):
        """Compute the multi-component Hungarian loss (and optionally matches).

        High-level steps:
        1. Extract and cast the ground-truth tensors from `y_true` in the
           order defined by `self.input_names`.
        2. For each enabled component, build a pairwise cost matrix between
           proposals and targets via `_calc_cost_matrix`.
        3. Combine component cost matrices into a single joint matrix using
           `add` and replace NaNs with a large constant cost.
        4. Run Hungarian matching on the joint cost matrix.
        5. Build per-component object masks and denominators used to
           normalize the summed losses.
        6. Aggregate the masked component losses using `add`.
        7. Optionally, construct and return matched `y_true`/`y_pred`
           tensors for each enabled component.

        Args:
            y_true: Dict mapping component names (e.g. "class", "bbox",
                "mask") to ground-truth tensors.
            y_pred: Dict (or ordered mapping) with the same set of component
                names mapping to prediction tensors.

        Returns:
            If `return_matching` is False, a tensor of per-sample scalar
            losses of shape `[batch]`.

            If `return_matching` is True, a tuple:

                `(matched_losses, y_true_output, y_pred_output)`

            where `matched_losses` has shape `[batch]`, and the dictionaries
            `y_true_output` and `y_pred_output` contain the matched tensors
            for each enabled component.
        """
        y_true = [tf.cast(y_true[input_name], tf.float32) for input_name in self.input_names]
        y_pred = list(y_pred.values())

        cost_matrices = [self._calc_cost_matrix(y_true_n, y_pred_n, **args) for y_true_n, y_pred_n, args in zip(y_true, y_pred, self.cost_matrix_args)]

        HM_cost_matrix = self.add(cost_matrices)
        HM_cost_matrix = tf.where(tf.math.is_nan(HM_cost_matrix), 2.0, HM_cost_matrix)


        idxs = self.HungarianMatching(HM_cost_matrix) #tf.stop_gradient()
        y_true_idxs, y_pred_idxs = idxs[0], idxs[1]

        unit_mask = tf.ones_like(y_true_idxs, dtype=tf.float32)
        objects_mask = tf.gather(y_true[0], y_true_idxs, batch_dims=1, axis=-1) if self.classification else unit_mask
        objects_masks = [objects_mask if masking else unit_mask for masking in self.masking_flags]
        denominators = [tf.reduce_sum(om, axis=-1) for om in objects_masks]

        matched_losses = self.add([tf.reduce_sum(self.extract_losses(cm, y_true_idxs, y_pred_idxs)*om, axis=-1)/d for cm, om, d in zip(cost_matrices, objects_masks, denominators)])

        if self.return_matching:
            objects_mask = tf.expand_dims(objects_mask, axis=-1)
            y_true_output = {name: self.extract_features(features, y_true_idxs, None, False, matching_args['perm'], matching_args['output_size']) for 
                             features, matching_args, name in zip(y_true, self.matching_args, self.input_names)}
            y_pred_output = {name: self.extract_features(features, y_pred_idxs, objects_mask, **matching_args) for 
                             features, matching_args, name in zip(y_pred, self.matching_args, self.input_names)}
            output = (matched_losses, y_true_output, y_pred_output)
        else:
            output = matched_losses
        return output


