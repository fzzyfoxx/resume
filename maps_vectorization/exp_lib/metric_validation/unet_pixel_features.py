import tensorflow as tf
import numpy as np
from models_src.Metrics import LossBasedMetric
from models_src.VecModels import AngleLoss, AngleLengthLoss
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--upload", default=0, type=int, help="Upload metrics to Mlflow run")
parser.add_argument("--run_id", default=None, type=str, help="Mlflow run_id")

kwargs = vars(parser.parse_args())

shape_class_metric = LossBasedMetric(tf.keras.losses.BinaryCrossentropy(name='LineDetLoss', reduction='sum_over_batch_size'))
angle_metric = LossBasedMetric(AngleLoss(gamma=1., name='LnAngleLoss', reduction='sum_over_batch_size'))
thickness_metric = LossBasedMetric(tf.keras.losses.MeanAbsoluteError(name='ThickLoss', reduction='sum_over_batch_size'))
center_vec_metric = LossBasedMetric(AngleLengthLoss(angle_gamma=1., length_gamma=1., dist_gamma=2., angle_weight=0.3, length_weight=0.7, dist_weight=1.0, name='CVLoss', reduction='sum_over_batch_size'))

@tf.function(input_signature=[tf.TensorSpec(shape=(cfg.test_batch_size,32,32,3), dtype=tf.float32),  # type: ignore
                            {
                                'shape_class': tf.TensorSpec(shape=(cfg.test_batch_size,32,32,3), dtype=tf.float32),  # type: ignore
                                'angle': tf.TensorSpec(shape=(cfg.test_batch_size,32,32,1), dtype=tf.float32),  # type: ignore
                                'thickness': tf.TensorSpec(shape=(cfg.test_batch_size,32,32,1), dtype=tf.float32), # type: ignore
                                'center_vec':tf.TensorSpec(shape=(cfg.test_batch_size,32,32,2), dtype=tf.float32) # type: ignore
                                },
                            {
                                'shape_class': tf.TensorSpec(shape=(cfg.test_batch_size,32,32), dtype=tf.float32),  # type: ignore
                                'angle': tf.TensorSpec(shape=(cfg.test_batch_size,32,32,1), dtype=tf.float32),  # type: ignore
                                'thickness': tf.TensorSpec(shape=(cfg.test_batch_size,32,32,1), dtype=tf.float32), # type: ignore
                                'center_vec':tf.TensorSpec(shape=(cfg.test_batch_size,32,32,1), dtype=tf.float32) # type: ignore
                                }
                              ], 
                              jit_compile=True, autograph=False)
def tf_eval(img, labels, weights):
    preds = trainer.model(img, training=False) # type: ignore

    shape_class_metric.update_state(labels['shape_class'], preds['shape_class'], weights['shape_class'])
    angle_metric.update_state(labels['angle'], preds['angle'], weights['angle'])
    thickness_metric.update_state(labels['thickness'], preds['thickness'], weights['thickness'])
    center_vec_metric.update_state(labels['center_vec'], preds['center_vec'], weights['center_vec'])

with tf.device('/GPU:0'):
    shape_class_metric.reset_state()
    angle_metric.reset_state()
    thickness_metric.reset_state()
    center_vec_metric.reset_state()

    pb = tf.keras.utils.Progbar(test_steps, stateful_metrics=['val_shape_class_loss','val_angle_loss', 'val_thickness_loss', 'val_center_vec_loss']) # type: ignore

    for i in range(test_steps): # type: ignore
        features, labels, labels_weights = next(test_iter) # type: ignore
        tf_eval(features, labels, labels_weights)
        pb.update(i+1, values=[('val_shape_class_loss', shape_class_metric.result().numpy()),
                               ('val_angle_loss', angle_metric.result().numpy()),
                               ('val_thickness_loss', thickness_metric.result().numpy()),
                               ('val_center_vec_loss', center_vec_metric.result().numpy())])
        
weighted_loss_value = np.mean([shape_class_metric.result().numpy(),angle_metric.result().numpy(),thickness_metric.result().numpy(),center_vec_metric.result().numpy()])
print('loss_value:', weighted_loss_value)


if kwargs['upload']:
    if kwargs['run_id'] is None:
        run_id = trainer.run_id # type: ignore
    else:
        run_id = kwargs['run_id']

    with mlflow.start_run(run_id=run_id): # type: ignore
        for metric_name, metric_value in zip(['val_shape_class_metric', 'val_angle_metric', 'val_thickness_metric', 'val_center_vec_metric'], \
                                             [m.result().numpy() for m in [shape_class_metric, angle_metric, thickness_metric, center_vec_metric]]):
            mlflow.log_metric(metric_name, metric_value) # type: ignore
