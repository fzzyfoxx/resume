import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from models_src.VecModels import flatten

def norm(x, reg=1.):
    return (x-tf.reduce_mean(x))/(tf.math.reduce_std(x)*reg+1e-4)

def norm_weights(w, batch_dims=0):
    return w/tf.cast(tf.reduce_prod(tf.shape(w)[batch_dims:]), w.dtype)

def weighted_sum(x, weights, axis=None, keepdims=False):
    return tf.reduce_sum(x*weights, axis=axis, keepdims=keepdims)

def weighted_std(x, weights, x_mean, axis=None, keepdims=None):
    return tf.reduce_sum(input_tensor=(x-x_mean)**2*weights, axis=axis, keepdims=keepdims)**0.5

def adaptive_loss_weights(weights, loss_values, reg=1.):

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
    def __init__(self, loss_func, reg=20., adapt_ratio=0.5, norm_clip=2., reduction='sum_over_batch_size', **kwargs):
        super().__init__(reduction=reduction,**kwargs)

        self.loss_func = loss_func
        self.reg = reg
        self.adapt_ratio = adapt_ratio
        self.norm_clip = norm_clip

    def call(self, y_true, y_pred):
        return self.loss_func.call(y_true, y_pred)
    
    def adaptive_loss_weights(self, weights, loss_values):

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
        losses = self.loss_func.call(y_true, y_pred)
        adapted_weight = self.adaptive_loss_weights(sample_weight, losses)

        return (1-self.adapt_ratio)*sample_weight + self.adapt_ratio*adapted_weight
    
    def __call__(self, y_true, y_pred, sample_weight=None):

        if sample_weight is not None:
            sample_weight = tf.stop_gradient(self.adapt_weights(y_true, y_pred, sample_weight))
        return super().__call__(y_true, y_pred, sample_weight)

class LossBasedMetric(tf.keras.metrics.Mean):
    def __init__(self, loss_func, **kwargs):
        super().__init__(**kwargs)

        self.loss_func = loss_func
        self.loss_func.reduction = tf.keras.losses.Reduction.NONE

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.loss_func(y_true, y_pred, sample_weight)
        values = tf.reduce_sum(values)/tf.cast(tf.reduce_prod(tf.shape(values)), tf.float32)
        return super().update_state(values, None)

class F12D(tf.keras.metrics.Metric):
    def __init__(self, threshold, name='F1', **kwargs):
        super(F12D, self).__init__(name=name, **kwargs)

        self.f1 = tf.keras.metrics.F1Score(threshold=threshold, average='micro')
        self.flatten = tf.keras.layers.Flatten()
        self.score = self.add_weight(name='f1', initializer='zeros')
        self.iterations = self.add_weight(name='iters', initializer='zeros')
        self.threshold = threshold

    def get_config(self):
        return {**super().get_config(), 'threshold': self.threshold}

    def update_state(self, y_true, y_pred, sample_weight=None):

        self.score.assign_add(self.f1(self.flatten(tf.cast(y_true, tf.float32)), self.flatten(y_pred)))
        self.iterations.assign_add(1.0)

    def result(self):
        return self.score/self.iterations
    
class WeightedF12D(tf.keras.metrics.Metric):
    def __init__(self, name='F1', threshold=0.5, average='micro', **kwargs):
        super().__init__(name=name, **kwargs)

        self.score = tf.keras.metrics.Mean()
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        prec = self.precision(y_true, y_pred, sample_weight)
        rec = self.recall(y_true, y_pred, sample_weight)

        score = tf.math.divide_no_nan(2*prec*rec, prec+rec)

        self.score.update_state(score)

    def reset_state(self):
        self.score.reset_state()
        self.precision.reset_state()
        self.recall.reset_state()

    def result(self):
        return self.score.result()

def extract_coords(bbox):
    return bbox[...,0], bbox[...,1], bbox[...,2], bbox[...,3]
    
def calc_area(Y1,X1,Y2,X2):
    return tf.abs(X2-X1)*tf.abs(Y2-Y1)

def IoU(a,b):
    aY1, aX1, aY2, aX2 = extract_coords(a)
    bY1, bX1, bY2, bX2 = extract_coords(b)

    xI = tf.abs(tf.clip_by_value(bX2, aX1, aX2) - tf.clip_by_value(bX1, aX1, aX2))
    yI = tf.abs(tf.clip_by_value(bY2, aY1, aY2) - tf.clip_by_value(bY1, aY1, aY2))

    I = xI*yI
    U = calc_area(aY1, aX1, aY2, aX2)+calc_area(bY1, bX1, bY2, bX2)-I

    return tf.math.divide_no_nan(I,U)

class MultiLevelFocalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, pooling_levels=2, alpha=0.75, gamma=2, reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(name="MLBFC",reduction=reduction,**kwargs)
        
        self.alpha = alpha
        self.gamma = gamma
        self.pooling_levels = pooling_levels
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.binary_focal_crossentropy(self.flatten(y_true), self.flatten(y_pred), apply_class_balancing=True, alpha=self.alpha, gamma=self.gamma)
        for i in range(self.pooling_levels):
            y_true = tf.nn.max_pool2d(y_true, 3, 2, 'VALID')
            y_pred = tf.nn.max_pool2d(y_pred, 3, 2, 'VALID')
            loss += tf.keras.losses.binary_focal_crossentropy(self.flatten(y_true), self.flatten(y_pred), apply_class_balancing=True, alpha=self.alpha, gamma=self.gamma)

        return loss/(1+self.pooling_levels)
    

class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, name='IoU'):
        super().__init__(name=name)

        self.score = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):

        scores = tf.expand_dims(IoU(y_true, y_pred), axis=-1)

        self.score.update_state(scores, sample_weight=sample_weight)

    def reset_state(self):
        self.score.reset_state()

    def result(self):
        return self.score.result()


class HungarianMatching:
    def __init__(self, matching_loss_func=tf.keras.losses.BinaryCrossentropy(), pool_size=4, perm=[0,2,1]):

        self.matching_loss = matching_loss_func
        self.matching_loss.reduction = tf.keras.losses.Reduction.NONE

        self.pool_size = pool_size
        self.perm = perm

    def _calc_losses(self, cost_matrix):
        match_idxs = tf.map_fn(lambda x: tf.transpose(tf.convert_to_tensor(linear_sum_assignment(x)), perm=[1,0]), elems=cost_matrix, fn_output_signature=tf.int32)
        return tf.gather_nd(cost_matrix, match_idxs, batch_dims=1)
    
    def _gen_matching(self, cost_matrix, y_true, y_pred):
        match_idxs = tf.map_fn(lambda x: tf.convert_to_tensor(linear_sum_assignment(x)), elems=cost_matrix, fn_output_signature=tf.int32)
        return tf.gather(tf.squeeze(y_true, axis=1), match_idxs[:,1], batch_dims=1, axis=1), \
            tf.gather(tf.squeeze(y_pred, axis=2), match_idxs[:,0], batch_dims=1, axis=1)

    @tf.function
    def __call__(self, y_true, y_pred):
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
    def __init__(self, hungarian_matching, loss_func=tf.keras.losses.BinaryCrossentropy(), reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(name="HL",reduction=reduction,**kwargs)
        
        self.hungarian_matching = hungarian_matching
        self.loss_func = loss_func

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        match_idxs = tf.stop_gradient(self.hungarian_matching(y_true, y_pred))
        y_true = tf.gather(y_true, match_idxs[:,:,1], batch_dims=1, axis=-1)
        y_pred = tf.gather(y_pred, match_idxs[:,:,0], batch_dims=1, axis=-1)
        return self.loss_func(y_true, y_pred)
    
class HungarianClassificationMetric(tf.keras.metrics.Metric):
    def __init__(self, hungarian_matching, metric_func, name='ClassMetric', **kwargs):
        super(HungarianClassificationMetric, self).__init__(name=name, **kwargs)

        self.hungarian_matching = hungarian_matching
        self.metric_func = metric_func
        #self.flatten = tf.keras.layers.Flatten()

        self.score = self.add_weight(name=name+'_score', initializer='zeros') #tf.Variable(0, dtype=tf.float32)#
        self.iterations = self.add_weight(name=name+'_iters', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        shape = tf.shape(y_true)
        B, H, W, T = shape[0], shape[1], shape[2], shape[3]

        match_idxs = tf.stop_gradient(self.hungarian_matching(y_true, y_pred))
        y_true = tf.reshape(tf.gather(y_true, match_idxs[:,:,1], batch_dims=1, axis=-1), (B, H*W*T))
        y_pred = tf.reshape(tf.gather(y_pred, match_idxs[:,:,0], batch_dims=1, axis=-1), (B, H*W*T))
        self.score.assign_add(self.metric_func(y_true, y_pred))
        self.iterations.assign_add(1.0)

    def reset_states(self):
        self.score.assign(0.0)
        self.iterations.assign(0.0)

    def result(self):
        return self.score/self.iterations
    

class MultivariantHungarianLoss():
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
        return 1-tf.clip_by_value(IoU(a,b), 1e-5, 1-1e-5)
    
    ### Bbox regression
    def Ln(self, a, b):
        return tf.reduce_mean(tf.abs((a-b)**self.bbox_reg_rank), axis=-1)
    
    ### Combined BBox Loss
    def BBoxLoss(self, a, b):
        return self.iou_weight*self.IoULoss(a,b) + (1-self.iou_weight)*self.Ln(a,b)
    

    ##### Hungarian Matching #####

    @tf.autograph.experimental.do_not_convert
    def HungarianMatching(self, cost_matrix):
        match_idxs = tf.map_fn(lambda x: tf.transpose(tf.convert_to_tensor(tf.numpy_function(linear_sum_assignment, [x], [tf.int64, tf.int64])), perm=[1,0]), elems=cost_matrix, fn_output_signature=tf.int64)
        y_true_idxs = match_idxs[...,1]
        y_pred_idxs = match_idxs[...,0]
        return y_true_idxs, y_pred_idxs
    
    def add(self, cost_matrices):
        cost_matrix = cost_matrices[0] * self.losses_weights[0]
        for i,cm in enumerate(cost_matrices[1:]):
            cost_matrix += cm * self.losses_weights[i+1]
        return cost_matrix
    
    @staticmethod
    def extract_losses(cost_matrix, y_true_idxs, y_pred_idxs):
        return tf.gather(tf.gather(cost_matrix, y_pred_idxs, batch_dims=1, axis=1), y_true_idxs, batch_dims=2, axis=2)
    

    def extract_features(self, features, idxs, mask, class_mask, perm, output_size):
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
    

