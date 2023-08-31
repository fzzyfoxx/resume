import tensorflow as tf
import keras_tuner as kt
import mlflow
from matplotlib import pyplot as plt
import cv2 as cv
import os
import time
import src.map_generator as mg
import numpy as np
from google.cloud import storage
import shutil

storage_client = storage.Client()


mlflow.tensorflow.autolog(log_datasets=False, log_models=False, disable=True)

class SmoothOutput(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(SmoothOutput, self).__init__(**kwargs)

        self.epsilon = epsilon

    def call(self, x, training=False):
        x = tf.clip_by_value(x, self.epsilon, 1-self.epsilon)
        return x
    

    
class LRCallback(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self,
                 warmup_lr=1e-5,
                 warmup_epochs=1,
                 initial_epoch=0,
                 scale2init_epoch=True,
                 decay_ratio=1.0,
                 decay_value=0.0,
                 **kwargs):
        super(LRCallback, self).__init__(schedule=self.schedule_func, **kwargs)
        self.initial_epoch = initial_epoch
        self.decay_ratio = decay_ratio
        self.decay_value = decay_value
        self.scale2init_epoch = scale2init_epoch
        self.warmup_lr = warmup_lr
        self.warmup_epochs = warmup_epochs

    def schedule_func(self, epoch, lr):
        if self.scale2init_epoch:
            epoch -= max(0,self.initial_epoch)

        if epoch==self.warmup_epochs:
            return self.init_lr
        elif epoch<=self.warmup_epochs-1:
            return self.warmup_lr
        else:
            return lr*self.decay_ratio-self.decay_value

    def on_train_begin(self, logs=None):
        self.init_lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        return super().on_train_begin(logs)
    

    
class BuildHyperModel(kt.HyperModel):
    def __init__(self, model_func, model_hp, optimizer_class, optimizer_hp, loss_class, loss_hp, metrics_classes, metrics_hp, mlflow_log=False, mlflow_instance=None, **kwargs):
        super(BuildHyperModel, self).__init__(self, **kwargs)

        self.model_func = model_func
        self.model_hp = model_hp

        self.optimizer = optimizer_class
        self.optimizer_hp = optimizer_hp

        self.loss = loss_class
        self.loss_hp = loss_hp

        self.metrics = metrics_classes
        self.metrics_hp = metrics_hp

        self.mlflow_log = mlflow_log
        self.mlflow = mlflow_instance

    def _format_hypermatameters(self, hp, hp_dict):
        hp_args = {}
        for key, args in hp_dict.items():
            if args['type']=='tuner':
                hp_args[key]=args['class'](hp, **args['args'])
            else:
                hp_args[key]=args['value']

        return hp_args

    def build(self, hp):
        model_hp_args = self._format_hypermatameters(hp, self.model_hp)

        model = self.model_func(model_hp_args)

        model.compile(optimizer=self.optimizer(**self._format_hypermatameters(hp, self.optimizer_hp)),
                      loss=self.loss(**self._format_hypermatameters(hp, self.loss_hp)),
                      metrics=[metric(**self._format_hypermatameters(hp, metric_hp)) for metric, metric_hp in zip(self.metrics,self.metrics_hp)])
        
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        callbacks = kwargs.get('callbacks')
        kwargs.pop('callbacks')
        
        callbacks += [callback(**self._format_hypermatameters(hp, callback_hp)) for callback, callback_hp in zip(kwargs.get('callbacks_classes'), kwargs.get('callbacks_hp'))]
        
        kwargs.pop('callbacks_classes')
        kwargs.pop('callbacks_hp')

        if self.mlflow_log:
            with self.mlflow.start_run():
                self.mlflow.log_params(hp.values)
                self.mlflow.tensorflow.autolog(log_datasets=False)
                return model.fit(*args, callbacks=callbacks, **kwargs)
        else:
            return model.fit(*args, callbacks=callbacks, **kwargs)
        

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(features, label):


    feature = {
        'features': _bytes_feature(tf.io.serialize_tensor(features)),
        'label': _bytes_feature(tf.io.serialize_tensor(label))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(features, label):
    tf_string = tf.py_function(serialize_example, [features, label], tf.string)
    return tf.reshape(tf_string, ())

feature_description = {
    'features': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string)
}

def _parse_function(example_proto, dtypes):
    features, label = tf.io.parse_single_example(example_proto, feature_description).values()

    features = tf.io.parse_tensor(features, dtypes[0])
    label = tf.io.parse_tensor(label, dtypes[1])

    return features, label

def create_path_if_needed(path):
    folders = os.path.split(path)
    curr_path = ''
    for folder in folders:
        curr_path = os.path.join(curr_path, folder)
        if not os.path.exists(curr_path):
            os.mkdir(curr_path)


class DatasetGenerator:
    def __init__(self, cfg, map_generator):
        self.cfg = cfg

        self.fmg = map_generator

        self.output_types = {
            '1': {'output': [tf.uint8, tf.int32], 'label_shape': (None, self.cfg.max_vertices_num*2)},
            '2': {'output': [tf.float32, tf.float32], 'label_shape': (self.cfg.target_size, self.cfg.target_size, 1)},
            '3': {'output': [tf.float32, tf.float32], 'label_shape': (self.cfg.target_size, self.cfg.target_size, None)}
        }

        self.map_decoder = mg.map_generator_decoder(cfg)

    @tf.function
    def _gen_images(self, *args):
        '''
            Uses full_map_generator_class to produce feature image and coresponding label

            In the second part shapes are set
        '''
        output_args = self.output_types[str(self.cfg.output_type)]
        features, label = tf.py_function(self.fmg.gen_full_map, [], output_args['output'])

        return features, label
    
    @tf.function
    def _set_shapes(self, features, label):
        # shapes definition
        output_args = self.output_types[str(self.cfg.output_type)]
        features.set_shape((self.cfg.target_size, self.cfg.target_size, 3))
        label.set_shape(output_args['label_shape'])

        return features, label
    
    def save_tfrec_dataset(self, folds_num=1, starting_num=0, ds_path='./datasets/train'):
        create_path_if_needed(ds_path)

        for fold in range(folds_num):
            self.fmg.reload_parcel_inputs()
            self.new_dataset(repeat=False, from_saved=False, batch=False)

            self.ds = self.ds.map(tf_serialize_example, self.cfg.num_parallel_calls)
            print(f'\n\033[1msaving fold {fold+1}/{folds_num}\033[0m')
            pb = tf.keras.utils.Progbar(self.cfg.fold_size)
            with tf.io.TFRecordWriter(f'{ds_path}/ds-{fold+starting_num}.tfrec') as writer:
                for feature, label in self.ds_iter:
                    writer.write(serialize_example(feature, label))
                    pb.add(1)

    def upload_dataset_to_storage(self, name, ds_path='./datasets/train'):
        bucket_name = storage_client.project + name
        ds_files = [(os.path.join(ds_path, filename), filename) for filename in os.listdir(ds_path)]

        # create if neede and get bucket
        try:
            bucket = storage_client.get_bucket(bucket_name)
        except:
            bucket = storage_client.create_bucket(storage_client.bucket(bucket_name), location='eu')

        pb = tf.keras.utils.Progbar(len(ds_files))
        for filepath, filename in ds_files:
            blob = bucket.blob(filename)
            blob.upload_from_filename(filepath)
            pb.add(1)

    def download_dataset_from_storage(self, name, ds_path='./datasets/train'):
        create_path_if_needed(ds_path)
        bucket_name = storage_client.project + name
        bucket = storage_client.get_bucket(bucket_name)
        blobs = [blob.name for blob in bucket.list_blobs()]
        ds_files = [(os.path.join(ds_path, filename), filename) for filename in blobs]

        pb = tf.keras.utils.Progbar(len(ds_files))
        for filepath, blob_name in ds_files:
            bucket.blob(blob_name).download_to_filename(filepath)
            pb.add(1)

    def delete_bucket(self, name):
        bucket_name = storage_client.project + name
        storage_client.get_bucket(bucket_name).delete(force=True)


    def new_dataset(self, repeat=True, from_saved=False, batch=True, ds_path='./datasets/train'):
        '''
            create new random dataset based on cfg parameters
            number of rows is defined by cfg.fold_size
            returns tf.data.Dataset object
        '''
        print('\n\033[1mGenerate new dataset\033[0m')
        if not from_saved:
            ds = tf.data.Dataset.range(self.cfg.fold_size)
            ds = ds.map(self._gen_images, num_parallel_calls=self.cfg.num_parallel_calls)
        else:
            output_types = self.output_types[str(self.cfg.output_type)]['output']
            ds_files = [os.path.join(ds_path, filename) for filename in os.listdir(ds_path)]
            ds = tf.data.TFRecordDataset(ds_files, num_parallel_reads=self.cfg.num_parallel_calls)
            ds = ds.map(lambda x: _parse_function(x, output_types), num_parallel_calls=self.cfg.num_parallel_calls)
        
        ds = ds.map(self._set_shapes, num_parallel_calls=self.cfg.num_parallel_calls)
        if batch:
            if self.cfg.output_type==1:
                ds = ds.padded_batch(self.cfg.ds_batch_size, padded_shapes=([self.cfg.target_size]*2+[3], [self.cfg.max_shapes_num, self.cfg.max_vertices_num*2]), padding_values=(np.uint8(255), 0))
            elif self.cfg.output_type==2:
                ds = ds.batch(self.cfg.ds_batch_size)
            elif self.cfg.output_type==3:
                ds = ds.padded_batch(self.cfg.ds_batch_size, padded_shapes=([self.cfg.target_size]*2+[3], [self.cfg.target_size]*2+[self.cfg.max_shapes_num]), padding_values=(0.0, 0.0))
        if repeat:
            ds = ds.repeat()
        self.ds = ds
        self.ds_iter = iter(ds)

    def dataset_speed_test(self,test_iters=20):
        print('\n\033[1mDataset generator speed test\033[0m')
        start_time = time.time()
        for _ in range(test_iters):
            _ = next(self.ds_iter)
        proc_time = time.time()-start_time

        print('time per batch: %.3fs | time per example: %.3fs' % (proc_time/test_iters,proc_time/(self.cfg.ds_batch_size*test_iters)))

    def decode_and_draw(self, img, label):
        decoded_shapes = self.map_decoder.decode_shape_output(label)
        decoded_img = self.map_decoder.decode_image(img)

        cv.polylines(decoded_img, decoded_shapes, False, (200,0,0), 5)
        plt.figure(figsize=(10,10))
        plt.imshow(decoded_img)

    def test_output(self,):
        print('\n\033[1mDataset output test\033[0m')
        output_type = self.cfg.output_type
        if output_type==1:
            img, label = next(self.ds_iter)
            print(img.shape, label.shape)
            self.decode_and_draw(img[0], label[0])
        elif output_type==2:
            img, label = next(self.ds_iter)
            print(img.shape, label.shape)

            plt.figure(figsize=(10,10))
            plt.imshow(img[0])
            plt.imshow(label[0], alpha=0.5, cmap='gray')

    @staticmethod
    def delete_dataset(path):
        os.system(f'rm -r {path}')

    @staticmethod
    def get_ds_sizes(path):
        ['{}: {:.3f} MB'.format(filename, os.path.getsize(os.path.join(path, filename))*1e-6) for filename in os.listdir(path)]