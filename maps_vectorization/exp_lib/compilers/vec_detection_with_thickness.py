import os
import sys
sys.path.append(os.path.abspath("../"))

import tensorflow as tf
from models_src.VecModels import NoSplitMixedBboxVecLoss
from models_src.Metrics import LossBasedMetric

def get_compile_args(
        gamma=1,
        loss_weights={'vecs': 0.6, 'class': 0.25, 'thickness': 0.15}
    ):
    compile_args = {
        'optimizer': tf.keras.optimizers.Adam(1e-3),
        'loss': {
            'vecs': NoSplitMixedBboxVecLoss(name='VecLoss', gamma=gamma, reduction='sum_over_batch_size'),
            'class': tf.keras.losses.BinaryCrossentropy(name='ClassLoss', reduction='sum_over_batch_size', axis=-1),
            'thickness': tf.keras.losses.MeanAbsoluteError(name='ThickLoss', reduction='sum_over_batch_size')
        },
        'loss_weights': loss_weights,
        'weighted_metrics': {'vecs': LossBasedMetric(NoSplitMixedBboxVecLoss(name='VecMixLoss', gamma=gamma, reduction='none'), name='VecMixMetric')}
    }

    return compile_args