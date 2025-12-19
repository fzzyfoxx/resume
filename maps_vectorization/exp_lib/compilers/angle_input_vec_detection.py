import os
import sys
sys.path.append(os.path.abspath("../"))

import tensorflow as tf
from models_src.VecModels import NoSplitMixedBboxVecMultiPropLoss, MultiSampleMAELoss, MultiSampleBinaryCrossEntropy, NoSplitMixedBboxVecMultiPropMetric
from models_src.Metrics import LossBasedMetric

def get_compile_args(
        gamma=1,
        conf_threshold=5,
        size=32,
        vec_loss_weight=0.6,
        sample_reduction_method='mean',
        loss_weights={'vecs': 0.6, 'class': 0.25, 'thickness': 0.15}
    ):
    compile_args = {
        'optimizer': tf.keras.optimizers.Adam(1e-3),
        'loss': {
            'vecs': NoSplitMixedBboxVecMultiPropLoss(name='VecLoss', conf_threshold=conf_threshold, size=size, gamma=1, vec_loss_weight=vec_loss_weight, norm=True, reduction='sum_over_batch_size'),
            'class': MultiSampleBinaryCrossEntropy(name='ClassLoss', sample_reduction_method=sample_reduction_method, reduction='sum_over_batch_size'),
            'thickness': MultiSampleMAELoss(name='ThickLoss', sample_reduction_method=sample_reduction_method, reduction='sum_over_batch_size')
        },
        'loss_weights': loss_weights,
        'weighted_metrics': {'vecs': NoSplitMixedBboxVecMultiPropMetric(size=size, gamma=gamma, norm=False, name='VecMixMetric')}
    }

    return compile_args