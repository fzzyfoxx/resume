import tensorflow as tf
from models_src.VecModels import WeightedPixelCrossSimilarityCrossentropy

def get_compile_args(
        learning_rate=1e-3,
        label_smoothing=0.1
    ):

    compile_args = {
                'optimizer': tf.keras.optimizers.Adam(learning_rate),
                #'loss': {'Dot_Similarity': vcm.PixelCrossSimilarityCrossentropy(label_smoothing=0.1, axis=-1, reduction='sum_over_batch_size')},
                'loss': {'Dot_Similarity': WeightedPixelCrossSimilarityCrossentropy(label_smoothing=label_smoothing, reduction='sum_over_batch_size')},
                #'metrics': {'Dot_Similarity': PixelSimilarityF1(skip_first_pattern=True, threshold=0.5)}
            }
    
    return compile_args