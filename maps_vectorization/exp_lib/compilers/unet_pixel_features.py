import tensorflow as tf
from models_src.Metrics import WeightedF12D
from models_src.VecModels import AngleLoss, AngleLengthLoss

def get_compile_args(
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
        'optimizer': tf.keras.optimizers.Adam(1e-3),
        'loss': {
            'shape_class': tf.keras.losses.BinaryCrossentropy(name='LineDetLoss', reduction='sum_over_batch_size'), 
            'angle': AngleLoss(gamma=gamma, name='LnAngleLoss', reduction='sum_over_batch_size'), 
            'thickness': tf.keras.losses.MeanAbsoluteError(name='ThickLoss', reduction='sum_over_batch_size'),
            'center_vec': AngleLengthLoss(angle_gamma=angle_gamma, length_gamma=length_gamma, dist_gamma=dist_gamma, angle_weight=angle_weight, 
                                          length_weight=length_weights, dist_weight=dist_weight, name='CVLoss', reduction='sum_over_batch_size')
            },
        'loss_weights': loss_weights,
        'weighted_metrics': None, #weighted_metrics,
        'metrics': {'shape_class': WeightedF12D()}
    }

    return compile_args