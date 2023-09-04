import tensorflow as tf
from scipy.optimize import linear_sum_assignment

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

        self.score.assign_add(self.f1(self.flatten(y_true), self.flatten(y_pred)))
        self.iterations.assign_add(1.0)

    def result(self):
        return self.score/self.iterations


class HungarianMatching:
    def __init__(self, matching_loss_func=tf.keras.losses.BinaryCrossentropy(), pool_size=4):

        self.matching_loss = matching_loss_func
        self.matching_loss.reduction = tf.keras.losses.Reduction.NONE

        self.pool_size = pool_size

    def _calc_losses(self, cost_matrix):
        match_idxs = tf.map_fn(lambda x: tf.transpose(tf.convert_to_tensor(linear_sum_assignment(x)), perm=[1,0]), elems=cost_matrix, fn_output_signature=tf.int32)
        return tf.gather_nd(cost_matrix, match_idxs, batch_dims=1)
    
    def _gen_matching(self, cost_matrix, y_true, y_pred):
        match_idxs = tf.map_fn(lambda x: tf.convert_to_tensor(linear_sum_assignment(x)), elems=cost_matrix, fn_output_signature=tf.int32)
        return tf.gather(tf.squeeze(y_true, axis=1), match_idxs[:,1], batch_dims=1, axis=1), \
            tf.gather(tf.squeeze(y_pred, axis=2), match_idxs[:,0], batch_dims=1, axis=1)

    @tf.function
    def __call__(self, y_true, y_pred):
        y_true_pooled = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=self.pool_size)(y_true)
        y_pred_pooled = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=self.pool_size)(y_pred)

        pooled_shape = tf.shape(y_true_pooled)
        B, H, W, T = pooled_shape[0], pooled_shape[1], pooled_shape[2], pooled_shape[3]
        P = tf.shape(y_pred_pooled)[-1]

        y_true_pooled = tf.expand_dims(tf.transpose(tf.reshape(y_true_pooled, (B,H*W,T)), perm=[0,2,1]), axis=1)
        y_pred_pooled = tf.expand_dims(tf.transpose(tf.reshape(y_pred_pooled, (B,H*W,P)), perm=[0,2,1]), axis=2)

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