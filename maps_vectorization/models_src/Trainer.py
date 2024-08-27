import os
import math
import matplotlib.pyplot as plt
import inspect
import keras_tuner as kt
import mlflow
import warnings
import tensorflow as tf
import numpy as np
import shutil
from google.cloud import storage


storage_client = storage.Client()

from models_src.UNet_model import UNet
from models_src.SegNet_model import SegNet
from models_src.DETR import DETRTransformer
from models_src.Deeplab import DeepLabModel

from models_src.Support import SmoothOutput, DatasetGenerator
from models_src.Trainer_support import BuildHyperModel

mlflow.tensorflow.autolog(log_datasets=False, log_models=False, disable=True)


class TrainingProcessor:
    def __init__(self, cfg, map_generator):
        self.cfg = cfg
        self.map_generator = map_generator

        self.model_gen_funcs = {
            'UNet': self._gen_UNet,
            'SegNet': self._gen_SegNet,
            'DETR': self._gen_DETR,
            'ResNet': self._gen_ResNet,
            'Deeplab': self._gen_Deeplab,
            'custom': self._gen_custom_model
        }

        self.steps_per_epoch = int(cfg.fold_size/cfg.ds_batch_size)

        self.cfg_attrs = self._get_cfg_attrs()

        self.reload_dataset(reload_parcels=False)


    def _get_cfg_attrs(self):
        return [(a, getattr(self.cfg, a)) for a in dir(self.cfg) if not a.startswith('__')]

    def _gen_UNet(self, model_args):
        x = UNet(**model_args, name='UNet')

        return x
    
    def _gen_Deeplab(self, model_args):
        x = DeepLabModel(**model_args)

        return x
    
    def _gen_SegNet(self, model_args):
        inputs = tf.keras.layers.Input((self.cfg.target_size,self.cfg.target_size,3), dtype=tf.float32, name='Map_Input')
        x = SegNet(**model_args, name='SegNet')(inputs)
        outputs = SmoothOutput()(x)
        return tf.keras.Model(inputs, outputs)
    
    def _gen_DETR(self, model_args):
        model = DETRTransformer(**model_args, name='DETR')
        model.build((None, self.cfg.target_size, self.cfg.target_size, 3))
        return model
    
    def load_model_generator(self, model_generator):
        self.model_generator = model_generator
    
    def _gen_ResNet(self, model_args):
        return self.model_generator(**model_args)
    
    def _gen_custom_model(self, model_args):
        return self.model_generator(**model_args)

    def reload_dataset(self, reload_parcels=True):
        if reload_parcels:
            self.map_generator.reload_parcel_inputs()
        self.dg = DatasetGenerator(self.cfg, self.map_generator)

    def _log_mlflow_params(self,):
        try:
            mlflow.start_run()
            print(f'MLflow run: {mlflow.active_run().info.run_name}')
            for key in self.model_args.keys():
                mlflow.log_param(key, self.model_args[key])

            for key, value in self.cfg_attrs:
                mlflow.log_param(key, value)

            for key in self.loss_args.keys():
                mlflow.log_param('loss.'+key, self.loss_args[key])

            mlflow.log_param('model_type', self.model_type)
        except:
            None

    def compile_model(self, model_type, model_args, optimizer, loss, metrics, loss_weights=None, print_summary=True, log=True, export_model=True, summary_kwargs={}):
        self.model_args = model_args
        self.model_type = model_type
        self.loss_args = loss.get_config()

        self.model = self.model_gen_funcs[model_type](model_args)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.compile(
                optimizer=optimizer,
                loss = loss,
                metrics = metrics,
                loss_weights = loss_weights
                )
        
        if log:
            self._log_mlflow_params()
            if export_model:
                mlflow.tensorflow.log_model(self.model, 'model',  custom_objects=self._get_custom_measures(loss, metrics))

        if print_summary:
            print(self.model.summary(**summary_kwargs))

        self.initial_epoch = 0

    def _get_custom_measures(self,loss, metrics):
        custom_objects = {}
        for c in ([loss]+metrics):
            if c.__module__.split('.')[0]!='keras':
                custom_objects[c.__class__.__name__] = c
        return custom_objects

    def train_model(self, epochs, callbacks=None, continue_run=False, export_final_state=True, export_model=True, validation_freq=0):
        if continue_run:
            self._log_mlflow_params()
            if export_model:
                mlflow.tensorflow.log_model(self.model, 'model', custom_objects=self._get_custom_measures(self.model.loss, self.model.metrics))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.dg.ds, 
                           validation_data=self.dg.val_ds if validation_freq>0 else None,
                           validation_steps=self.dg.val_steps if validation_freq>0 else None,
                           validation_freq=validation_freq,
                           epochs=epochs+self.initial_epoch, 
                           steps_per_epoch=self.dg.train_steps, 
                           callbacks=callbacks, 
                           initial_epoch=self.initial_epoch)
        self.initial_epoch += epochs

        if export_final_state:
            #save weights
            self.model.save_weights('./final_state/variables')
            mlflow.log_artifacts('./final_state', 'final_state')
        
        try:
            mlflow.end_run()
        except:
            None

    ### LOAD MODEL

    def _load_mlflow_args(self, run_args):
        cfg_params = {}
        # fit method parameters
        fit_args = ['params.'+p for p in inspect.getfullargspec(tf.keras.Model.fit)[0]]
        run_args = run_args[~run_args.index.str.contains('opt_', case=False) & run_args.index.str.contains('params.') & ~run_args.index.isin(fit_args)]

        # get loss attributes
        loss_args = {key.split('.')[-1]: value for key, value in run_args[run_args.index.str.contains('loss.')].to_dict().items()}
        run_args = run_args[~run_args.index.str.contains('loss.')]

        # look for general configuration parameters
        for key, _ in self.cfg_attrs:
            search_results = [(p,run_args[p]) for p in run_args.index if 'params.'+key==p]
            if len(search_results)>0:
                param, value = search_results[0]
                cfg_params[key] = getattr(__builtins__, type(getattr(self.cfg,key)).__name__)(value)
                run_args = run_args.drop(param)

        # get model args
        model_args = {key.split('.')[-1]: value for key, value in run_args.to_dict().items()}

        if not np.all([getattr(self.cfg, key)==value for key, value in cfg_params.items()]):
            for key, value in cfg_params.items():
                setattr(self.cfg, key, value)

            self.reload_dataset()

        self.loss_args = loss_args
        self.model_args = model_args

        print('\n\033[1mLoss params\033[0m')
        print(loss_args)

        print('\n\033[1mModel params\033[0m')
        print(model_args)

    def save_temp_weights(self, weights_path):
        try:
            os.mkdir(weights_path)
        except:
            None
        self.model.save_weights(f'./{weights_path}/weights.keras', save_format='keras')

    def load_temp_weights(self, weights_path, skip_mismatch=True, by_name=True):
        self.model.load_weights(f'./{weights_path}/weights.keras', skip_mismatch, by_name)

    def load_model(self, run_name, log=True, load_final_state=True):
        run_args = mlflow.search_runs(filter_string=f"tags.mlflow.runName like '{run_name}%'").iloc[0]
        self.initial_epoch = int(run_args['params.epochs']) #int(run_args['params.initial_epoch']) + 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = mlflow.tensorflow.load_model(os.path.join(run_args.artifact_uri, "model"))

        if load_final_state:
            mlflow.artifacts.download_artifacts(run_id=run_args.run_id,artifact_path='final_state',
                                    dst_path='./')
            self.model.load_weights('final_state/variables')

        if log:
            self._load_mlflow_args(run_args)
            self._log_mlflow_params()
            mlflow.tensorflow.log_model(self.model, 'model', custom_objects=self._get_custom_measures(self.model.loss, self.model.metrics))

    ### HYPERPARAMETER TUNING

    def build_tuner(self, model_type, model_hp, optimizer_class, optimizer_hp, loss_class, loss_hp, metrics_classes, metrics_hp, mlflow_log, mlflow_instance, tuner_class, tuner_args):
        self.tuner = tuner_class(
            BuildHyperModel(self.model_gen_funcs[model_type], model_hp, optimizer_class, optimizer_hp, loss_class, loss_hp, metrics_classes, metrics_hp, mlflow_log, mlflow_instance),
            **tuner_args
        )

    def run_tuner(self, epochs, callbacks=[], callbacks_classes=[], callbacks_hp=[], validation_freq=1):
        self.tuner.search(self.dg.ds, 
                           validation_data=self.dg.val_ds if validation_freq>0 else None,
                           validation_steps=self.dg.val_steps if validation_freq>0 else None,
                           validation_freq=validation_freq,
                           epochs=epochs, 
                           steps_per_epoch=self.dg.train_steps,
                           callbacks=callbacks, 
                           callbacks_classes=callbacks_classes, 
                           callbacks_hp=callbacks_hp
        )
        print('Best Parameters')
        print(self.tuner.get_best_hyperparameters()[0].values) 

        self.model = self.tuner.get_best_models(num_models=1)[0]

    def upload_tensorboard_logs(self, filename ,folder_path='logs', delete_after=True):
        shutil.make_archive(filename, 'zip', folder_path)

        bucket = storage_client.get_bucket(storage_client.project + '-tb-logs')
        result = bucket.blob(filename).upload_from_filename(filename+'.zip')

        if delete_after:
            os.remove(filename+'.zip')

        return result
    
    def download_tensorboard_logs(self, filename, save_path='', unpack_path='logs', delete_after=True):
        bucket = storage_client.get_bucket(storage_client.project + '-tb-logs')
        result = bucket.blob(filename).download_to_filename(os.path.join(save_path, filename+'.zip'))

        shutil.unpack_archive(os.path.join(save_path, filename+'.zip'), unpack_path, 'zip')

        if delete_after:
            os.remove(os.path.join(save_path, filename+'.zip'))

        return result

    ### DISPLAY RESULTS

    def print_predicted_maps(self, background_type='features', cmap='gray', alpha=0.5 ,threshold=0.5):

        features, labels = next(self.dg.ds_iter)
        preds = self.model.predict(features)
        preds = tf.where(preds>threshold, 1.0, 0.0)

        cols=4
        rows = math.ceil(len(features)/cols)

        if background_type=='features':
            backgrounds = features
        elif background_type=='labels':
            backgrounds = labels
        else:
            backgrounds = [None]*len(preds)

        fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*4))
        for background, pred, ax in zip(backgrounds, preds, axs.flat):
            if type(background).__name__!='NoneType':
                ax.imshow(background)
            ax.imshow(pred, cmap=cmap, alpha=alpha)



class TrainingProcessor2:
    def __init__(self, cfg, mlflow_instance=None):
        self.cfg = cfg
        self.cfg_attrs = self._get_cfg_attrs()

        self.mlflow = mlflow_instance if mlflow_instance is not None else mlflow

    def load_dataset(self, ds, train_steps, val_ds=None, val_steps=None):
        self.ds = ds
        self.train_steps = train_steps
        self.val_ds = val_ds
        self.val_steps = val_steps

        self.cfg.train_samples = train_steps*self.cfg.train_batch_size
        self.cfg.val_samples = val_steps*self.cfg.val_batch_size

    def _get_cfg_attrs(self):
        return [(a, getattr(self.cfg, a)) for a in dir(self.cfg) if not a.startswith('__')]
    
    def load_model_generator(self, model_generator):
        self.model_generator = model_generator
        self.cfg.generator_name = model_generator.__name__
    
    def _gen_custom_model(self, model_args):
        return self.model_generator(**model_args)

    def _log_mlflow_params(self):
        
        for key in self.model_args.keys():
            self.mlflow.log_param(key, self.model_args[key])

        for key, value in self.cfg_attrs:
            self.mlflow.log_param(key, value)

        for name, loss_args in self.loss_args:
            for key, value in loss_args.items():
                self.mlflow.log_param('loss.'+name+'_'+key, value)

        #mlflow.log_param('model_type', self.model_type)

    def compile_model(self, model_args, optimizer, loss, metrics=None, weighted_metrics=None, loss_weights=None, print_summary=True, summary_kwargs={}):
        self.model_args = model_args
        #self.model_type = model_type
        self.loss_args = [(k, v.get_config()) for k, v in  loss.items()]

        self.model = self._gen_custom_model(model_args)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.compile(
                optimizer=optimizer,
                loss = loss,
                metrics = metrics,
                loss_weights = loss_weights,
                weighted_metrics = weighted_metrics
                )

        if print_summary:
            print(self.model.summary(**summary_kwargs))

        self.initial_epoch = 0

    def _get_custom_measures(self,loss, metrics):
        if metrics is None:
            metrics = {}
            
        custom_objects = {}
        for c in {**loss, **(metrics if metrics.__class__ == dict else dict([(metric.name, metric) for metric in metrics]))}.values():
            if c.__module__.split('.')[0]!='keras':
                custom_objects[c.__class__.__name__] = c
        return custom_objects

    def train_model(self, epochs, log=True, callbacks=None, export_final_state=True, export_model=True, validation_freq=1):

        if log:
            #self.mlflow.tensorflow.autolog(log_datasets=False, log_models=True, disable=False, checkpoint=False, log_every_epoch=True)
            run = self.mlflow.start_run()
            print(f'MLflow run: {run.info.run_name}')
            mlflow_callback = [mlflow.keras.MlflowCallback(run)]
            callbacks = mlflow_callback if callbacks is None else callbacks+mlflow_callback
        else:
            None
            self.mlflow.tensorflow.autolog(log_datasets=False, log_models=False, disable=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.ds, 
                           validation_data=self.val_ds if (validation_freq>0) & (self.val_ds is not None) else None,
                           validation_steps=self.val_steps if (validation_freq>0) & (self.val_steps is not None) else None,
                           validation_freq=validation_freq,
                           epochs=epochs+self.initial_epoch, 
                           steps_per_epoch=self.train_steps, 
                           callbacks=callbacks, 
                           initial_epoch=self.initial_epoch)

        if log:
            with self.mlflow.active_run():
                self.mlflow.log_params({
                    'epoch': epochs,
                    'initial_epoch': self.initial_epoch
                })
                self._log_mlflow_params()
                if export_final_state:
                    #save weights
                    self._prepare_path('./final_state/temp')
                    self.model.save_weights('./final_state/temp/final_state.weights.h5')
                    self.mlflow.log_artifacts('./final_state/temp', '')
                if export_model:
                    self.mlflow.tensorflow.log_model(self.model, 'model',  custom_objects=self._get_custom_measures(self.model.loss, self.model.metrics))
        
        try:
            self.mlflow.end_run()
        except:
            None

        self.initial_epoch += epochs


    ### LOAD MODEL

    def _load_mlflow_args(self, run_args):
        cfg_params = {}
        # fit method parameters
        fit_args = ['params.'+p for p in inspect.getfullargspec(tf.keras.Model.fit)[0]]
        run_args = run_args[~run_args.index.str.contains('opt_', case=False) & run_args.index.str.contains('params.') & ~run_args.index.isin(fit_args)]

        # get loss attributes
        loss_args = {key.split('.')[-1]: value for key, value in run_args[run_args.index.str.contains('loss.')].to_dict().items()}
        run_args = run_args[~run_args.index.str.contains('loss.')]

        # look for general configuration parameters
        for key, _ in self.cfg_attrs:
            search_results = [(p,run_args[p]) for p in run_args.index if 'params.'+key==p]
            if len(search_results)>0:
                param, value = search_results[0]
                cfg_params[key] = getattr(__builtins__, type(getattr(self.cfg,key)).__name__)(value)
                run_args = run_args.drop(param)

        # get model args
        model_args = {key.split('.')[-1]: value for key, value in run_args.to_dict().items()}

        if not np.all([getattr(self.cfg, key)==value for key, value in cfg_params.items()]):
            for key, value in cfg_params.items():
                setattr(self.cfg, key, value)


        self.loss_args = loss_args
        self.model_args = model_args

        print('\n\033[1mLoss params\033[0m')
        print(loss_args)

        print('\n\033[1mModel params\033[0m')
        print(model_args)

    @staticmethod
    def _prepare_path(path):
        try:
            os.makedirs(path, exist_ok=True)
        except:
            None

    def save_temp_weights(self, weights_path, filename='final_state', use_model_name=True):

        if use_model_name:
            weights_path = os.path.join(weights_path, self.model.name)

        self._prepare_path(weights_path)
        self.model.save_weights(f'./{weights_path}/{filename}.weights.h5')

    def load_temp_weights(self, weights_path, skip_mismatch=True):
        self.model.load_weights(f'./{weights_path}.weights.h5', skip_mismatch=skip_mismatch)

    def load_mlflow_weights(self, run_name, weights_path='./', skip_mismatch=True):
        run_args = self.mlflow.search_runs(filter_string=f"tags.mlflow.runName like '{run_name}%'").iloc[0]
        #self.initial_epoch = int(run_args['params.epochs'])

        self._download_and_load_mlflow_weights(weights_path, run_args, skip_mismatch)

    def _download_and_load_mlflow_weights(self, weights_path, run_args, skip_mismatch=True):
        self._prepare_path(weights_path)
        self.mlflow.artifacts.download_artifacts(run_id=run_args.run_id,artifact_path='final_state.weights.h5',
                                    dst_path=weights_path)
        self.model.load_weights(os.path.join(weights_path,'final_state.weights.h5'), skip_mismatch=skip_mismatch)

    def load_model(self, run_name, weights_path='./final_state/temp', load_final_state=True):
        run_args = self.mlflow.search_runs(filter_string=f"tags.mlflow.runName like '{run_name}%'").iloc[0]
        self.initial_epoch = int(run_args['params.epochs']) #int(run_args['params.initial_epoch']) + 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = self.mlflow.tensorflow.load_model(os.path.join(run_args.artifact_uri, "model"))

        if load_final_state:
            self._download_and_load_mlflow_weights(weights_path, run_args, skip_mismatch=False)

        self._load_mlflow_args(run_args)