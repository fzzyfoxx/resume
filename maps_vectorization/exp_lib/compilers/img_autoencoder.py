import tensorflow as tf

def get_compile_args():
    compile_args = {
        'optimizer': tf.keras.optimizers.Adam(1e-3),
        'loss': {
            'target_img': tf.keras.losses.MeanAbsoluteError(name='AEloss', reduction='sum_over_batch_size'), 
            },
        'loss_weights': None,
        'weighted_metrics': None, #weighted_metrics,
        'metrics': None
    }

    return compile_args