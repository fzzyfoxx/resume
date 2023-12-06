import tensorflow as tf
from matplotlib import pyplot as plt
import cv2 as cv
import os
import time
import src.map_generator as mg
import numpy as np
from google.cloud import storage
import warnings
import math

storage_client = storage.Client()


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
    
        

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(names, inputs):

    '''feature = {
        'features': _bytes_feature(tf.io.serialize_tensor(features)),
        'label': _bytes_feature(tf.io.serialize_tensor(label))
    }'''
    feature = {name: _bytes_feature(tf.io.serialize_tensor(x)) for name, x in zip(names, inputs)}

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(features, label):
    tf_string = tf.py_function(serialize_example, [features, label], tf.string)
    return tf.reshape(tf_string, ())

'''feature_description = {
    'features': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string)
}'''

def _parse_function(example_proto, dtypes, feature_description):
    inputs = tf.io.parse_single_example(example_proto, feature_description).values()

    '''features = tf.io.parse_tensor(features, dtypes[0])
    label = tf.io.parse_tensor(label, dtypes[1])'''

    inputs = [tf.io.parse_tensor(x, x_type) for x, x_type in zip(inputs, dtypes)]

    return inputs

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

        img_shape = (self.cfg.target_size, self.cfg.target_size, 3)

        self.output_types = {
            '1': {'output': [tf.uint8, tf.int32], 
                  'input_shapes': [img_shape, (None, self.cfg.max_vertices_num*2)],
                  'feature_names': ['features', 'label']
                  },
            '2': {'output': [tf.float32, tf.float32], 
                  'input_shapes': [img_shape, (self.cfg.target_size, self.cfg.target_size, 1)],
                  'feature_names': ['features', 'label']
                  },
            '3': {'output': [tf.float32, tf.bool], 
                  'input_shapes': [img_shape, (self.cfg.target_size, self.cfg.target_size, None)],
                  'feature_names': ['features', 'label']
                  },
            '4': {'output': [tf.float32, tf.float32], 
                  'input_shapes': [img_shape, (4,)],
                  'feature_names': ['features', 'label']
                  },
            '5': {'output': [tf.float32, tf.float32], 
                  'input_shapes': [img_shape, (None,4)],
                  'feature_names': ['Afeatures', 'Bbbox']
                  },
            '6': {'output': [tf.float32, tf.float32, tf.bool], 
                  'input_shapes': [img_shape, (None, 4), (self.cfg.target_size, self.cfg.target_size, None)],
                  'feature_names': ['Afeatures', 'Bbbox','Cmask']
                  },
            '7': {'output': [tf.float32, tf.bool], 
                  'input_shapes': [img_shape, (self.cfg.target_size, self.cfg.target_size,1)],
                  'feature_names': ['features', 'label']
                  },
            '8': {'output': [tf.float32, tf.bool], 
                  'input_shapes': [img_shape, (self.cfg.target_size, self.cfg.target_size, None)],
                  'feature_names': ['Afeatures', 'Cmask']
                  },
            '9': {'output': [tf.float32, tf.float32], 
                  'input_shapes': [img_shape, img_shape],
                  'feature_names': ['features', 'label']
                  },
            '10': {'output': [tf.float32, tf.bool, tf.bool], 
                  'input_shapes': [img_shape, (None, self.cfg.target_size, self.cfg.target_size, 1), (self.cfg.target_size, self.cfg.target_size, None)],
                  'feature_names': ['Afeatures', 'Bclusters', 'Cmask']
                  },
            '11': {'output': [tf.float32, tf.bool], 
                  'input_shapes': [img_shape, (self.cfg.target_size, self.cfg.target_size, 5)],
                  'feature_names': ['Afeatures', 'Cmask']
                  },
            '12': {'output': [tf.float32, tf.float32], 
                  'input_shapes': [img_shape, (self.cfg.target_size, self.cfg.target_size, None)],
                  'feature_names': ['Afeatures', 'Cmask']
                  },
            '13': {'output': [tf.float32, tf.float32], 
                  'input_shapes': [img_shape, (self.cfg.target_size, self.cfg.target_size, 1)],
                  'feature_names': ['Afeatures', 'Cmask']
                  },
            '14': {'output': [tf.float32, tf.float32], 
                  'input_shapes': [img_shape, (None, 4)],
                  'feature_names': ['Afeatures', 'Bbbox']
                  },
            'pmg1': {'output': [tf.float32, tf.bool], 
                  'input_shapes': [(self.cfg.target_size, self.cfg.target_size*2, 3), (self.cfg.target_size, self.cfg.target_size*2)],
                  'feature_names': ['Afeatures', 'Cmask']
                  }
        }

        self.map_decoder = mg.map_generator_decoder(cfg)

    @tf.function
    def _gen_images(self, *args):
        '''
            Uses full_map_generator_class to produce feature image and coresponding label

            In the second part shapes are set
        '''
        output_args = self.output_types[str(self.cfg.output_type)]
        inputs = tf.py_function(self.fmg.gen_full_map, [], output_args['output'])

        return inputs
    
    @tf.function
    def _set_shapes(self, *args):
        # shapes definition
        inputs = args
        output_args = self.output_types[str(self.cfg.output_type)]

        for input, input_shape in zip(inputs, output_args['input_shapes']):
            input.set_shape(input_shape)

        return inputs
    
    def _gen_feature_description(self):
        return {name: tf.io.FixedLenFeature([], tf.string) for name in self.output_types[str(self.cfg.output_type)]['feature_names']}
    
    def save_tfrec_dataset(self, folds_num=1, starting_num=0, ds_path='./datasets/train'):
        create_path_if_needed(ds_path)
        names = self.output_types[str(self.cfg.output_type)]['feature_names']
        for fold in range(folds_num):
            self.fmg.reload_parcel_inputs()
            self.new_dataset(repeat=False, from_saved=False, batch=False, format_output=False)

            #self.ds = self.ds.map(tf_serialize_example, self.cfg.num_parallel_calls)
            print(f'\n\033[1msaving fold {fold+1}/{folds_num}\033[0m')
            pb = tf.keras.utils.Progbar(self.cfg.fold_size)
            with tf.io.TFRecordWriter(f'{ds_path}/ds-{fold+starting_num}.tfrec') as writer:
                for inputs in self.ds_iter:
                    writer.write(serialize_example(names,inputs))
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

    def download_dataset_from_storage(self, name, storage_client, ds_path='./datasets/train'):
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


    def new_dataset(self, repeat=True, from_saved=False, batch=True, format_output=True, validation=False, val_idxs=[], shuffle_buffer_size=0, fold_size=512, ds_path='./datasets/train'):
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
            feature_description = self._gen_feature_description()
            output_types = self.output_types[str(self.cfg.output_type)]['output']
            ds_files = [os.path.join(ds_path, filename) for i, filename in enumerate(os.listdir(ds_path)) if (i in val_idxs if validation else i not in val_idxs)]
            ds = tf.data.TFRecordDataset(ds_files, num_parallel_reads=self.cfg.num_parallel_calls)
            ds = ds.map(lambda x: _parse_function(x, output_types, feature_description), num_parallel_calls=self.cfg.num_parallel_calls)
        
        # set dataset shapes
        ds = ds.map(self._set_shapes, num_parallel_calls=self.cfg.num_parallel_calls)

        # format output
        if format_output:
            if self.cfg.output_type==5:
                ds = ds.map(lambda *x: (x[0], {'class': tf.ones((len(x[1]),)), 'bbox': x[1]}), num_parallel_calls=self.cfg.num_parallel_calls)

            elif self.cfg.output_type==6:
                ds = ds.map(lambda *x: (x[0], {'class': tf.ones((tf.shape(x[1])[-1],)), 'bbox': x[1], 'mask': x[2]}), num_parallel_calls=self.cfg.num_parallel_calls)

            elif self.cfg.output_type==8:
                ds = ds.map(lambda *x: (x[0], {'class': tf.ones((tf.shape(x[1])[-1],)), 'mask': x[1]}), num_parallel_calls=self.cfg.num_parallel_calls)

            elif self.cfg.output_type==10:
                ds = ds.map(lambda *x: (x[0][tf.newaxis]*tf.cast(x[1], tf.float32), {'class': tf.ones((tf.shape(x[2])[-1],)), 'mask': x[2]}), num_parallel_calls=self.cfg.num_parallel_calls)

            elif self.cfg.output_type==12:
                ds = ds.map(lambda *x: (x[0], {'class': tf.ones((tf.shape(x[1])[-1],)), 'mask': x[1]}), num_parallel_calls=self.cfg.num_parallel_calls)

            elif self.cfg.output_type==14:
                ds = ds.map(lambda *x: (x[0], {'class': tf.ones((tf.shape(x[1])[0],)), 'bbox': x[1]}), num_parallel_calls=self.cfg.num_parallel_calls)
            

        # batch and padding definitions
        if batch:
            if self.cfg.output_type==1:
                ds = ds.padded_batch(self.cfg.ds_batch_size, padded_shapes=([self.cfg.target_size]*2+[3], [self.cfg.max_shapes_num, self.cfg.max_vertices_num*2]), padding_values=(np.uint8(255), 0))
            
            elif self.cfg.output_type==2:
                ds = ds.batch(self.cfg.ds_batch_size)
            
            elif self.cfg.output_type==3:
                ds = ds.padded_batch(self.cfg.ds_batch_size, padded_shapes=([self.cfg.target_size]*2+[3], [self.cfg.target_size]*2+[self.cfg.max_shapes_num]), padding_values=(0.0, False))
            
            elif self.cfg.output_type==4:
                ds = ds.batch(self.cfg.ds_batch_size)

            elif self.cfg.output_type==5:
                ds = ds.padded_batch(self.cfg.ds_batch_size, padded_shapes=([self.cfg.target_size]*2+[3], 
                                                                            {'class': [self.cfg.max_shapes_num],
                                                                            'bbox': [self.cfg.max_shapes_num,4]}), 
                                    padding_values=(0.0, {'class': 0.0, 'bbox': 0.0}))
            
            elif self.cfg.output_type==6:
                ds = ds.padded_batch(self.cfg.ds_batch_size, padded_shapes=([self.cfg.target_size]*2+[3], 
                                                                            {'class': [self.cfg.max_shapes_num],
                                                                            'bbox': [self.cfg.max_shapes_num,4],
                                                                            'mask':[self.cfg.target_size]*2+[self.cfg.max_shapes_num]}), 
                                    padding_values=(0.0, {'class': 0.0, 'bbox': 0.0, 'mask': False}))
            elif self.cfg.output_type==7:
                ds = ds.batch(self.cfg.ds_batch_size)

            elif self.cfg.output_type==8:
                ds = ds.padded_batch(self.cfg.ds_batch_size, padded_shapes=([self.cfg.target_size]*2+[3], 
                                                                            {'class': [self.cfg.max_shapes_num],
                                                                            'mask':[self.cfg.target_size]*2+[self.cfg.max_shapes_num]}), 
                                    padding_values=(0.0, {'class': 0.0, 'mask': False}))
                
            elif self.cfg.output_type==9:
                ds = ds.batch(self.cfg.ds_batch_size)

            elif self.cfg.output_type==10:
                ds = ds.padded_batch(self.cfg.ds_batch_size, padded_shapes=([self.cfg.n_clusters]+[self.cfg.target_size]*2+[3], 
                                                                            {'class': [self.cfg.max_shapes_num],
                                                                            'mask':[self.cfg.target_size]*2+[self.cfg.max_shapes_num]}), 
                                    padding_values=(0.0, {'class': 0.0, 'mask': False}))
                
            elif self.cfg.output_type==11:
                ds = ds.batch(self.cfg.ds_batch_size)

            elif self.cfg.output_type==12:
                ds = ds.padded_batch(self.cfg.ds_batch_size, padded_shapes=([self.cfg.target_size]*2+[3], 
                                                                            {'class': [self.cfg.max_shapes_num],
                                                                            'mask':[self.cfg.target_size]*2+[self.cfg.max_shapes_num]}), 
                                    padding_values=(0.0, {'class': 0.0, 'mask': 0.0}))
            
            elif self.cfg.output_type==13:
                ds = ds.batch(self.cfg.ds_batch_size)

            elif self.cfg.output_type==14:
                ds = ds.padded_batch(self.cfg.ds_batch_size, padded_shapes=([self.cfg.target_size]*2+[3], 
                                                                            {'class': [self.cfg.max_shapes_num],
                                                                            'bbox': [self.cfg.max_shapes_num,4]}), 
                                    padding_values=(0.0, {'class': 0.0, 'bbox': 0.0}))
                
            elif self.cfg.output_type=='pmg1':
                ds = ds.batch(self.cfg.ds_batch_size)

        if repeat:
            ds = ds.repeat()
        
        if not validation:
            if shuffle_buffer_size>0:
                ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
            self.ds = ds
            self.ds_iter = iter(ds)
            self.train_steps = fold_size//(self.cfg.ds_batch_size if batch else 1)*len(ds_files) if from_saved else self.cfg.fold_size
        else:
            self.val_ds = ds
            self.val_iter = iter(ds)
            self.val_steps = fold_size//(self.cfg.ds_batch_size if batch else 1)*len(ds_files) if from_saved else self.cfg.fold_size

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


def plot_bbox_preds(features, labels, preds, target_size=256, plotsize=8, max_plots=None, cols=1, draw_preds=True, fontsize=3):
    plots_num = min(len(features), max_plots) if max_plots else len(features)
    rows = math.ceil(plots_num/cols)
    #decode features
    features = (features.numpy()*255).astype(np.uint8)
    #decode bboxes
    true_bboxes = (labels['bbox'].numpy()*target_size).astype(np.int32)
    pred_bboxes = (preds['bbox'].numpy()*target_size).astype(np.int32)
    if 'class' in preds.keys():
        probs = preds['class']
    else:
        probs = tf.zeros(pred_bboxes.shape[:-1])[...,tf.newaxis]

    fig, axs = plt.subplots(rows, cols, figsize=(plotsize*cols, plotsize*rows))

    for img, inst_true_bboxes, inst_pred_bboxes, inst_probs, i in zip(features, true_bboxes, pred_bboxes, probs, range(plots_num)):
        pred_img = img.copy()
        for true_bbox, pred_bbox, prob in zip(inst_true_bboxes, inst_pred_bboxes, inst_probs):
            if np.sum(true_bbox)>0:
                cv.rectangle(img, true_bbox[:2][::-1], true_bbox[2:][::-1], (255,0,0), 1)
            if np.sum(pred_bbox)>0:
                cv.rectangle(pred_img, pred_bbox[:2][::-1], pred_bbox[2:][::-1], (0,0,255), 1)
                if draw_preds:
                    cv.putText(pred_img, '{:.1f}%'.format(prob), pred_bbox[:2][::-1], cv.FONT_HERSHEY_SIMPLEX , fontsize, (0,0,255))

        img = cv.addWeighted(img, 0.5, pred_img, 0.5, 0)
        if plots_num==1:
            axs.imshow(img)
        else:
            axs.flat[i].imshow(img)

def plot_mask_preds(features, labels, preds, threshold=0.5, target_size=256, plotsize=6, cmap='gray', max_plots=None, alpha=0.5):
    rows = min(len(features), max_plots) if max_plots else len(features)

    features = (features.numpy()*255).astype(np.uint8)
    true_masks = (labels['mask'].numpy()*255).astype(np.uint8)[...,np.newaxis]
    pred_masks = (preds['mask'].numpy()*255).astype(np.uint8)[...,np.newaxis]

    fig, axs = plt.subplots(rows, 4, figsize=(4*plotsize, rows*plotsize))

    for img, inst_true_mask, inst_pred_mask, row in zip(features, true_masks, pred_masks, range(rows)):
        inst_true_mask = np.reshape(inst_true_mask, (-1, target_size, target_size, 1))
        inst_pred_mask = np.reshape(inst_pred_mask, (-1, target_size, target_size, 1))

        axs[row, 0].imshow(np.sum(inst_true_mask, axis=0), cmap=cmap)
        axs[row, 0].set_title('True Mask SUM', fontsize=8)

        axs[row, 1].imshow(np.sum(inst_pred_mask, axis=0), cmap=cmap)
        axs[row, 1].set_title('Pred Mask SUM', fontsize=8)

        binary_pred_mask = np.sum(np.where(inst_pred_mask>threshold, 0, 1), axis=0)
        axs[row, 2].imshow(binary_pred_mask, cmap=cmap)
        axs[row, 2].set_title('Binary Pred Mask SUM', fontsize=8)

        axs[row, 3].imshow(img)
        axs[row, 3].imshow(binary_pred_mask, cmap='gray', alpha=alpha)
        axs[row, 3].set_title('Pred Mask SUM on map', fontsize=12)




class MiniShapeGenerator:
    def __init__(self,
                max_colors = 10,
                max_points = 6,
                max_thickness = 3,
                max_scale = 5,
                window_size = 64,
                add_background_mask = False,
                color_mask = False
                ):
        
        for key, value in locals().items():
            setattr(self, key, value)

    @staticmethod
    def mini_shape_generation(window_size=16, lines_range=(1,6), line_points_range=(2,6), thickness_range=(1,3), scale_range=(2,5), color_mask=False):
        scale = np.random.randint(*scale_range)
        lines_num = np.random.randint(*lines_range)
        points_num = np.random.randint(*line_points_range)
        colors = np.reshape(np.random.randint(1, 250, lines_num*3, dtype=np.uint8), (lines_num, 3)).astype(np.float32)
        thickness = np.random.randint(*thickness_range, lines_num)
        
        #backgrund_color = np.random.randint(230,250)
        #img = np.ones((window_size*(2**scale), window_size*(2**scale), 3), np.uint8)*backgrund_color
        img = np.reshape(np.random.uniform(230,250, (window_size*(2**scale))**2*3), (window_size*(2**scale), window_size*(2**scale), 3)).astype(np.uint8)
        masks = []
        if color_mask:
            label_masks = []
        for th, clr in zip(thickness, colors):
            points = np.reshape(np.random.randint(0, window_size*2**scale, points_num*2), (points_num, 1, 2))
            img = cv.polylines(img, [points], False, clr.tolist(), th)
            mask = np.zeros((window_size*(2**scale), window_size*(2**scale), 1), np.uint8)
            clr_mask = cv.polylines(mask, [points], False, 1, th)
            clr_mask = tf.keras.layers.MaxPool2D(pool_size=2**scale)(clr_mask[tf.newaxis])
            if color_mask:
                label_mask = clr_mask * tf.constant(clr, tf.uint8)[tf.newaxis, tf.newaxis, tf.newaxis]
                label_masks.append(label_mask[0].numpy())
            masks.append(clr_mask[0].numpy())

        img2 = cv.resize(img, (window_size, window_size), interpolation=cv.INTER_AREA)

        if not color_mask:
            masks = [m-m*np.max(masks[i+1:], axis=0) for i,m in enumerate(masks[:-1])]+[masks[-1]]
        else:
            masks = [m-m*np.max(masks[i+1:], axis=0) for i,m in enumerate(label_masks[:-1])]+[label_masks[-1]]
        masks = np.stack(masks, axis=0)

        return img, img2, masks
    
    def reload_parcel_inputs(self, ):
        None
    
    def gen_full_map(self, ):
        _, img2, mask = self.mini_shape_generation(window_size=self.window_size, lines_range=(1,self.max_colors+1), line_points_range=(2,self.max_points+1), thickness_range=(1,self.max_thickness+1), scale_range=(2,self.max_scale+1), color_mask=self.color_mask)
        features = tf.constant(img2/255, tf.float32)
        #mask = tf.cast(tf.stack(mask, axis=0), tf.float32)
        #padding = self.max_colors-len(mask)
        #mask_mask = tf.concat([tf.ones((len(mask)+(1 if self.add_background_mask else 0))), tf.zeros((padding,))], axis=0)
        if not self.color_mask:
            mask = tf.cast(mask, tf.float32)
            background_mask = 1-tf.reduce_max(mask, axis=0, keepdims=True)
            mask = tf.concat([mask]+ ([background_mask] if self.add_background_mask else []), axis=0)
            mask = tf.transpose(tf.squeeze(mask, axis=-1), perm=[1,2,0])
        else:
            mask = tf.reduce_max(mask, axis=0)
            mask = tf.where(mask==0, 255, mask)
            mask = tf.cast(mask, tf.float32)/255

        return features, mask
    
    def gen_dataset(self, batch_size, examples_num, from_saved=False, repeat=True, format_output=True, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

        if not from_saved:
            ds = tf.data.Dataset.range(examples_num)
            ds = ds.map(lambda x: tf.py_function(self.gen_full_map, [], [tf.float32, tf.float32]), num_parallel_calls=4)
        if format_output:
            ds = ds.map(lambda *x: (x[0], {'class': tf.ones((tf.shape(x[1])[-1],)), 'mask': x[1]}), num_parallel_calls=4)
        if batch_size>0:
            #ds = ds.batch(batch_size)
            if not self.color_mask:
                pad_size = self.max_colors + (1 if self.add_background_mask else 0)
                ds = ds.padded_batch(batch_size, padded_shapes=([self.window_size]*2+[3], 
                                                                                {'class': [pad_size],
                                                                                'mask':[self.window_size]*2+[pad_size]}), 
                                        padding_values=(0.0, {'class': 0.0, 'mask': 0.0}))
            else:
                ds = ds.batch(batch_size)
        if repeat:
            ds = ds.repeat()

        self.ds = ds
        self.ds_iter = iter(ds)
    
    def plot_example(self, examples=1):
        imgs, labels = next(self.ds_iter)

        masks = tf.transpose(labels['mask'], perm=[0,3,1,2])
        mask_masks = labels['class']

        for img, mask, mask_mask, _ in zip(imgs, masks, mask_masks, range(examples)):
            masks_num = int(np.sum(mask_mask))
            fig, ax = plt.subplots(1,1, figsize=(4,4))
            ax.imshow(1-img)
            fig, axs = plt.subplots(1, masks_num, figsize=(4*masks_num, 4))
            for i, ax in enumerate(axs.flat):
                ax.imshow(1-img, cmap='gray')
                ax.imshow(mask[i], alpha=0.5, cmap='gray')
        plt.show()


class WeightedMaskME(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, L=1, name='wMSE', reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super(WeightedMaskME, self).__init__(**kwargs)

        self.alpha = alpha
        self.L = L
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true, y_pred):
        diff = y_pred-y_true
        over_est = tf.nn.relu(diff)**self.L*self.alpha
        under_est = tf.nn.relu(-diff)**self.L
        diff = self.flatten(over_est+under_est)
        diff = tf.reduce_mean(diff)
        return diff