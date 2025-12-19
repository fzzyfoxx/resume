import os
import sys
sys.path.append(os.path.abspath("../"))

import tensorflow as tf
from models_src.VecModels import NoSplitMixedBboxVecLoss
from models_src.Metrics import LossBasedMetric

def get_compile_args(
        gamma=1,
        size=32,
        loss_weights={'vecs': 0.5, 'conf': 0.15, 'class': 0.2, 'thickness': 0.15}
    ):
    compile_args = {
        'optimizer': tf.keras.optimizers.Adam(1e-3),
        'loss': {
            'vecs': NoSplitMixedBboxVecLoss(name='VecLoss', gamma=gamma, norm=True, size=size, reduction='sum_over_batch_size'),
            'conf': tf.keras.losses.BinaryCrossentropy(name='ConfLoss', reduction='sum_over_batch_size', axis=-1),
            'class': tf.keras.losses.BinaryCrossentropy(name='ClassLoss', reduction='sum_over_batch_size', axis=-1),
            'thickness': tf.keras.losses.MeanAbsoluteError(name='ThickLoss', reduction='sum_over_batch_size')
        },
        'loss_weights': loss_weights,
        'weighted_metrics': {'vecs': LossBasedMetric(NoSplitMixedBboxVecLoss(name='VecMixLoss', gamma=gamma, reduction='none'), name='VecMixMetric')}
    }

    return compile_args