import tensorflow as tf
from models_src.Metrics import LossBasedMetric
from models_src.VecModels import NoSplitMixedBboxVecLoss
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--upload", default=0, type=int, help="Upload metrics to Mlflow run")
parser.add_argument("--run_id", default=None, type=str, help="Mlflow run_id")

kwargs = vars(parser.parse_args())

vec_mixed_metric = LossBasedMetric(NoSplitMixedBboxVecLoss(name='VecMixLoss', gamma=1, reduction='none'))
vec_metric = LossBasedMetric(NoSplitMixedBboxVecLoss(name='VecLoss', gamma=1, reduction='none'))
bbox_metric = LossBasedMetric(NoSplitMixedBboxVecLoss(name='BBoxLoss', gamma=1, reduction='none'))
class_metric = LossBasedMetric(tf.keras.losses.BinaryCrossentropy(name='ClassLoss', reduction='none', axis=-1))
thickness_metric = LossBasedMetric(tf.keras.losses.MeanAbsoluteError(name='ThickLoss', reduction='none'))

@tf.function(input_signature=[tf.TensorSpec(shape=(cfg.test_batch_size,32,32,3), dtype=tf.float32),  # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,cfg.sample_points,2), dtype=tf.int32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,cfg.sample_points,4,2), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,cfg.sample_points,3), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,cfg.sample_points,1), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,cfg.sample_points), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(), dtype=tf.int32),
                              ], 
                              jit_compile=True, autograph=False)
def tf_eval(img, sample_points, vecs_label, class_label, thickness_label, class_weights, n):
    vecs_pred, class_pred, thickness_pred = trainer.model({'img': img, 'sample_points': sample_points}, training=False).values() # type: ignore
    vec_mixed_metric.update_state(vecs_label, vecs_pred, class_weights)
    class_metric.update_state(class_label, class_pred, class_weights)
    thickness_metric.update_state(thickness_label, thickness_pred, tf.expand_dims(class_weights, axis=-1))

    class_idx = tf.argmax(class_label, axis=-1)
    vecs_mask = tf.where(class_idx==2, 1., 0.)
    bbox_mask = tf.where(class_idx==1, 1., 0.)

    B = tf.shape(img)[0]
    vecs_num = tf.reduce_sum(vecs_mask, axis=None, keepdims=True)
    vecs_weights = vecs_mask*tf.math.divide_no_nan(tf.cast(n*B, tf.float32), vecs_num)

    bbox_num = tf.reduce_sum(bbox_mask, axis=None, keepdims=True)
    bbox_weights = bbox_mask*tf.math.divide_no_nan(tf.cast(n*B, tf.float32), bbox_num)

    vecs_pred, bbox_pred = tf.split(vecs_pred, 2, axis=-2)
    vecs_label, bbox_label = tf.split(vecs_label, 2, axis=-2)

    vec_metric.update_state(vecs_label, vecs_pred, vecs_weights)
    bbox_metric.update_state(bbox_label, bbox_pred, bbox_weights)

with tf.device('/GPU:0'):
    vec_mixed_metric.reset_state()
    class_metric.reset_state()
    pb = tf.keras.utils.Progbar(test_steps, stateful_metrics=['VecMixLoss','VecLoss', 'BBoxLoss', 'ClassLoss'])  # type: ignore

    for i in range(test_steps):   # type: ignore
        features, labels, labels_weights = next(test_iter) # type: ignore
        img, sample_points = features.values()
        vecs_labels, components_class, thickness_labels = labels.values()
        vecs_weights, class_weights, thickness_weights = labels_weights.values()
        tf_eval(img, sample_points, vecs_labels, components_class, thickness_labels, vecs_weights, cfg.sample_points) # type: ignore
        pb.update(i+1, values=[('VecMixLoss', vec_mixed_metric.result().numpy()),
                               ('VecLoss', vec_metric.result().numpy()),
                               ('BBoxLoss', bbox_metric.result().numpy()),
                               ('ClassLoss', class_metric.result().numpy()),
                               ('ThickLoss', thickness_metric.result().numpy())])
        
if kwargs['upload']:
    if kwargs['run_id'] is None:
        run_id = trainer.run_id # type: ignore
    else:
        run_id = kwargs['run_id']

    with mlflow.start_run(run_id=run_id): # type: ignore
        for metric_name, metric_value in zip(['VecMixedLoss', 'VecLoss', 'BBoxLoss', 'ClassLoss', 'ThickLoss'], \
                                             [m.result().numpy() for m in [vec_mixed_metric, vec_metric, bbox_metric, class_metric, thickness_metric]]):
            mlflow.log_metric(metric_name, metric_value) # type: ignore