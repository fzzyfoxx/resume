import tensorflow as tf
from models_src.Metrics import LossBasedMetric
from models_src.VecModels import NoSplitMixedBboxVecLoss
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--upload", default=0, type=int, help="Upload metrics to Mlflow run")
parser.add_argument("--run_id", default=None, type=str, help="Mlflow run_id")

kwargs = vars(parser.parse_args())

size = trainer.model_args['size'] # type: ignore

vec_mixed_metric = LossBasedMetric(NoSplitMixedBboxVecLoss(name='VecMixLoss', gamma=1, reduction='none'))
vec_metric = LossBasedMetric(NoSplitMixedBboxVecLoss(name='VecLoss', gamma=1, reduction='none'))
bbox_metric = LossBasedMetric(NoSplitMixedBboxVecLoss(name='BBoxLoss', gamma=1, reduction='none'))
conf_metric = LossBasedMetric(tf.keras.losses.BinaryCrossentropy(name='ConfLoss', reduction='none', axis=-1))
class_metric = LossBasedMetric(tf.keras.losses.BinaryCrossentropy(name='ClassLoss', reduction='none', axis=-1))
thickness_metric = LossBasedMetric(tf.keras.losses.MeanAbsoluteError(name='ThickLoss', reduction='none'))

@tf.function(input_signature=[tf.TensorSpec(shape=(cfg.test_batch_size,size,size,3), dtype=tf.float32),  # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,1), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,size,size,4,2), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,size,size,1), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,size,size,3), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,size,size,1), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,size,size,1), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,size,size), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,size,size), dtype=tf.float32), # type: ignore
                              tf.TensorSpec(shape=(cfg.test_batch_size,size,size), dtype=tf.float32) # type: ignore
                              ], 
                              jit_compile=True, autograph=False)
def tf_eval(img, angle_input, vec_label, conf_label, shape_class, thickness_label, vec_weights, conf_weights, class_weights, thickness_weights):
    preds = trainer.model({'img': img, 'angle_input': angle_input}, training=False) # type: ignore
    vec_pred, conf_pred, class_pred, thickness_pred = preds['vecs'], preds['conf'], preds['class'], preds['thickness']
    vec_mixed_metric.update_state(vec_label, vec_pred, vec_weights)
    conf_metric.update_state(conf_label, conf_pred, conf_weights)
    class_metric.update_state(shape_class, class_pred, class_weights)
    thickness_metric.update_state(thickness_label, thickness_pred, thickness_weights)

    class_idx = tf.argmax(shape_class, axis=-1)[...,tf.newaxis]
    all_shapes_mask = tf.where(vec_weights>0, 1., 0.)
    vecs_mask = tf.where(class_idx==2, 1., 0.) * all_shapes_mask
    bbox_mask = tf.where(class_idx==1, 1., 0.) * all_shapes_mask

    B = tf.shape(img)[0]
    S = tf.cast(tf.shape(img)[1]**2 * B, tf.float32)
    vecs_num = tf.reduce_sum(vecs_mask, axis=None, keepdims=True)
    vecs_weights = vecs_mask*tf.math.divide_no_nan(S, vecs_num)

    bbox_num = tf.reduce_sum(bbox_mask, axis=None, keepdims=True)
    bbox_weights = bbox_mask*tf.math.divide_no_nan(S, bbox_num)

    vec_metric.update_state(vec_label, vec_pred, vecs_weights)
    bbox_metric.update_state(vec_label, vec_pred, bbox_weights)

with tf.device('/GPU:0'):
    vec_mixed_metric.reset_state()
    vec_metric.reset_state()
    bbox_metric.reset_state()
    conf_metric.reset_state()
    class_metric.reset_state()
    thickness_metric.reset_state()

    pb = tf.keras.utils.Progbar(test_steps, stateful_metrics=['VecMixLoss','VecLoss', 'BBoxLoss', 'ConfLoss', 'ClassLoss', 'ThickLoss'])  # type: ignore

    for i in range(test_steps):   # type: ignore
        inputs, labels, weights = next(test_iter) # type: ignore
        img, angle_input = inputs['img'], inputs['angle_input']
        vec_label, conf_label, shape_class, thickness_label = labels['vecs'], labels['conf'], labels['class'], labels['thickness']
        vec_weights, conf_weights, class_weights, thickness_weights = weights['vecs'], weights['conf'], weights['class'], weights['thickness']
        tf_eval(img, angle_input, vec_label, conf_label, shape_class, thickness_label, vec_weights, conf_weights, class_weights, thickness_weights) # type: ignore
        pb.update(i+1, values=[('VecMixLoss', vec_mixed_metric.result().numpy()),
                               ('VecLoss', vec_metric.result().numpy()),
                               ('BBoxLoss', bbox_metric.result().numpy()),
                               ('ConfLoss', conf_metric.result().numpy()),
                               ('ClassLoss', class_metric.result().numpy()),
                               ('ThickLoss', thickness_metric.result().numpy())])
        
if kwargs['upload']:
    if kwargs['run_id'] is None:
        run_id = trainer.run_id # type: ignore
    else:
        run_id = kwargs['run_id']

    with mlflow.start_run(run_id=run_id): # type: ignore
        for metric_name, metric_value in zip(['VecMixLoss','VecLoss', 'BBoxLoss', 'ConfLoss', 'ClassLoss', 'ThickLoss'], \
                                             [m.result().numpy() for m in [vec_mixed_metric, vec_metric, bbox_metric, conf_metric, class_metric, thickness_metric]]):
            mlflow.log_metric(metric_name, metric_value) # type: ignore