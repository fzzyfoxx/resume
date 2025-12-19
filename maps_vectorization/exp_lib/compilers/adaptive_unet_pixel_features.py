import tensorflow as tf
from models_src.Metrics import WeightedF12D, AdaptiveWeightsLoss
from models_src.VecModels import AngleLoss, AngleLengthLoss

class mse(tf.keras.losses.Loss):
    def __init__(self, reduction='sum_over_batch_size', name='MSE', **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)

    def call(self, y_pred, y_true):
        return tf.reduce_mean(tf.abs(y_pred-y_true), axis=-1, keepdims=True)
    
class al(tf.keras.losses.Loss):
    def __init__(self, gamma=1., name='AL', reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)

        self.gamma = gamma

    def get_config(self):
        return {
            'name': self.name,
            'reduction': self.reduction,
            'gamma': self.gamma
        }
    
    def call(self, y_true, y_pred):

        return tf.abs(y_pred-y_true)

def get_compile_args(
        reg=20.,
        adapt_ratio=0.5,
        norm_clip=2.,
        gamma=1.,
        angle_gamma=1.,
        length_gamma=1.,
        dist_gamma=2.,
        angle_weight=0.3,
        length_weights=0.7,
        dist_weight=1.,
        loss_weights={'shape_class': 0.25, 'angle': 0.25, 'thickness': 0.25, 'center_vec': 0.25}
):
    compile_args = {
        'optimizer': tf.keras.optimizers.SGD(1e-3),
        'loss': {
            'shape_class': AdaptiveWeightsLoss(tf.keras.losses.BinaryCrossentropy(), reg=reg, adapt_ratio=adapt_ratio, norm_clip=norm_clip, name='LineDetLoss', reduction='sum_over_batch_size'), 
            'angle': AdaptiveWeightsLoss(AngleLoss(reduction='none'), reg=reg, adapt_ratio=adapt_ratio, norm_clip=norm_clip, name='LnAngleLoss', reduction='sum_over_batch_size'), 
            'thickness': AdaptiveWeightsLoss(mse(reduction='none'), reg=reg, adapt_ratio=adapt_ratio, norm_clip=norm_clip, name='ThickLoss', reduction='sum_over_batch_size'),
            'center_vec': AdaptiveWeightsLoss(AngleLengthLoss(angle_gamma=angle_gamma, length_gamma=length_gamma, dist_gamma=dist_gamma, angle_weight=angle_weight, 
                                          length_weight=length_weights, dist_weight=dist_weight, reduction='none'), reg=reg, adapt_ratio=adapt_ratio, norm_clip=norm_clip, name='CVLoss', reduction='sum_over_batch_size')
            },
        'loss_weights': loss_weights,
        'weighted_metrics': None, #weighted_metrics,
        'metrics': {'shape_class': WeightedF12D()}
    }

    return compile_args

'''
AngleLengthLoss(angle_gamma=angle_gamma, length_gamma=length_gamma, dist_gamma=dist_gamma, angle_weight=angle_weight, 
                                          length_weight=length_weights, dist_weight=dist_weight, reduction='none')
'''