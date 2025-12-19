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
import json
from google.cloud import storage

storage_client = storage.Client()

from models_src.UNet_model import UNet
from models_src.SegNet_model import SegNet
from models_src.DETR import DETRTransformer
from models_src.Deeplab import DeepLabModel

from models_src.Support import SmoothOutput, DatasetGenerator
from models_src.Trainer_support import BuildHyperModel
from exp_lib.utils.load_mlflow_model import download_mlflow_model_weights

mlflow.tensorflow.autolog(log_datasets=False, log_models=False, disable=True)


class TrainingProcessor:
    """Legacy training helper that wraps model creation, compilation, training,
    MLflow logging, dataset handling and simple visualization.

    This version is tightly coupled to ``DatasetGenerator`` and several
    predefined model builders (UNet, SegNet, DETR, DeepLab, etc.).

    Attributes
    ----------
    cfg : Any
        Configuration object with all hyperparameters and dataset settings.
    map_generator : Any
        Object responsible for providing parcel / map inputs used by
        ``DatasetGenerator``.
    model_gen_funcs : dict[str, Callable]
        Mapping from model type name to factory method.
    steps_per_epoch : int
        Number of steps per training epoch based on configuration.
    cfg_attrs : list[tuple[str, Any]]
        Cached list of configuration attribute names and values used for
        logging / restoring from MLflow runs.
    dg : DatasetGenerator
        Dataset generator used for training and validation.
    model : tf.keras.Model
        Compiled Keras model (after ``compile_model`` is called).
    initial_epoch : int
        Epoch index from which to resume training.
    """

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
        """Return a list of configuration attributes for logging.

        Returns
        -------
        list[tuple[str, Any]]
            Pairs of attribute name and value for all public attributes
            of ``self.cfg``.
        """
        return [(a, getattr(self.cfg, a)) for a in dir(self.cfg) if not a.startswith('__')]

    def _gen_UNet(self, model_args):
        """Instantiate a UNet model with the given arguments.

        Parameters
        ----------
        model_args : dict
            Keyword arguments forwarded to :class:`UNet`.

        Returns
        -------
        tf.keras.Model
            Constructed UNet model.
        """
        x = UNet(**model_args, name='UNet')

        return x
    
    def _gen_Deeplab(self, model_args):
        """Instantiate a DeepLab model with the given arguments.

        Parameters
        ----------
        model_args : dict
            Keyword arguments forwarded to :class:`DeepLabModel`.

        Returns
        -------
        tf.keras.Model
            Constructed DeepLab model.
        """
        x = DeepLabModel(**model_args)

        return x
    
    def _gen_SegNet(self, model_args):
        """Instantiate a SegNet model wrapped with ``SmoothOutput``.

        The input shape is derived from ``self.cfg.target_size``.

        Parameters
        ----------
        model_args : dict
            Keyword arguments forwarded to :class:`SegNet`.

        Returns
        -------
        tf.keras.Model
            Keras model from input to smoothed output.
        """
        inputs = tf.keras.layers.Input((self.cfg.target_size,self.cfg.target_size,3), dtype=tf.float32, name='Map_Input')
        x = SegNet(**model_args, name='SegNet')(inputs)
        outputs = SmoothOutput()(x)
        return tf.keras.Model(inputs, outputs)
    
    def _gen_DETR(self, model_args):
        """Instantiate and build a DETR transformer model.

        Parameters
        ----------
        model_args : dict
            Keyword arguments forwarded to :class:`DETRTransformer`.

        Returns
        -------
        tf.keras.Model
            Built DETR transformer model.
        """
        model = DETRTransformer(**model_args, name='DETR')
        model.build((None, self.cfg.target_size, self.cfg.target_size, 3))
        return model
    
    def load_model_generator(self, model_generator):
        """Register an external model generator callable.

        This is used by ``_gen_ResNet`` or ``_gen_custom_model`` to create
        models that are not hard-coded in this module.

        Parameters
        ----------
        model_generator : Callable
            Function or callable that returns a compiled or uncompiled
            ``tf.keras.Model`` when called with ``model_args``.
        """
        self.model_generator = model_generator
    
    def _gen_ResNet(self, model_args):
        """Instantiate a ResNet-like model via the registered generator.

        Parameters
        ----------
        model_args : dict
            Arguments forwarded to the registered ``model_generator``.

        Returns
        -------
        tf.keras.Model
            Model returned by ``model_generator``.
        """
        return self.model_generator(**model_args)
    
    def _gen_custom_model(self, model_args):
        """Instantiate a custom model via the registered generator.

        Parameters
        ----------
        model_args : dict
            Arguments forwarded to the registered ``model_generator``.

        Returns
        -------
        tf.keras.Model
            Model returned by ``model_generator``.
        """
        return self.model_generator(**model_args)

    def reload_dataset(self, reload_parcels=True):
        """(Re)create the :class:`DatasetGenerator`.

        Optionally forces the ``map_generator`` to reload its parcel inputs
        before creating a new dataset generator.

        Parameters
        ----------
        reload_parcels : bool, default True
            If ``True``, call ``self.map_generator.reload_parcel_inputs()``
            before instantiating ``DatasetGenerator``.
        """
        if reload_parcels:
            self.map_generator.reload_parcel_inputs()
        self.dg = DatasetGenerator(self.cfg, self.map_generator)

    def _log_mlflow_params(self,):
        """Log configuration, model, and loss parameters to MLflow.

        This method assumes that ``self.model_args``, ``self.loss_args`` and
        ``self.cfg_attrs`` have been populated (typically by
        :meth:`compile_model` or :meth:`load_model`). If MLflow is not
        available or logging fails, the exception is silently ignored.
        """
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
        """Create and compile a model of the given ``model_type``.

        Parameters
        ----------
        model_type : str
            Key in ``self.model_gen_funcs`` specifying which constructor to
            use (e.g. ``'UNet'``, ``'SegNet'``).
        model_args : dict
            Arguments passed to the selected model constructor.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer instance to use.
        loss : tf.keras.losses.Loss
            Primary loss instance used during training.
        metrics : list[tf.keras.metrics.Metric]
            List of metrics to track during training.
        loss_weights : Optional[list[float]]
            Optional per-output loss weights.
        print_summary : bool, default True
            If ``True``, print ``model.summary()`` after compilation.
        log : bool, default True
            If ``True``, log hyperparameters and (optionally) the model
            definition to MLflow.
        export_model : bool, default True
            If ``True`` and ``log`` is enabled, log the compiled model to
            MLflow.
        summary_kwargs : dict, optional
            Extra keyword arguments forwarded to ``model.summary``.
        """
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
        """Return a dict of custom loss/metric classes for MLflow logging.

        Only non-keras (i.e. user-defined) losses/metrics are included so
        they can be reconstructed when loading from MLflow.

        Parameters
        ----------
        loss : tf.keras.losses.Loss
            Loss instance used by the model.
        metrics : list[tf.keras.metrics.Metric]
            Metrics used by the model.

        Returns
        -------
        dict[str, type]
            Mapping from class name to class object for custom components.
        """
        custom_objects = {}
        for c in ([loss]+metrics):
            if c.__module__.split('.')[0]!='keras':
                custom_objects[c.__class__.__name__] = c
        return custom_objects

    def train_model(self, epochs, callbacks=None, continue_run=False, export_final_state=True, export_model=True, validation_freq=0):
        """Train the compiled model on the current dataset.

        Parameters
        ----------
        epochs : int
            Number of epochs to train for (added to ``self.initial_epoch``).
        callbacks : list[tf.keras.callbacks.Callback], optional
            Optional list of callbacks passed to ``model.fit``.
        continue_run : bool, default False
            If ``True``, assume an existing MLflow run and log parameters and
            model again before training.
        export_final_state : bool, default True
            If ``True``, save final weights to ``./final_state`` and log them
            as MLflow artifacts.
        export_model : bool, default True
            If ``True`` and ``continue_run`` is ``True``, log the model to
            MLflow before training.
        validation_freq : int, default 0
            Frequency (in epochs) at which to run validation. If 0, no
            validation dataset is used.
        """
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
        """Parse MLflow run parameters and update configuration/model args.

        This reads back configuration values, loss parameters and model
        arguments from an MLflow ``run_args`` row (as returned by
        ``mlflow.search_runs``). If configuration values differ from the
        current ``self.cfg``, the config is updated and the dataset is
        reloaded.

        Parameters
        ----------
        run_args : pandas.Series
            MLflow run information for a single run.
        """
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
        """Save model weights to a temporary local directory.

        The weights are stored under ``./{weights_path}/weights.keras``.

        Parameters
        ----------
        weights_path : str
            Directory path (relative or absolute) where weights will be
            written. The directory is created if it does not exist.
        """
        try:
            os.mkdir(weights_path)
        except:
            None
        self.model.save_weights(f'./{weights_path}/weights.keras', save_format='keras')

    def load_temp_weights(self, weights_path, skip_mismatch=True, by_name=True):
        """Load model weights from a temporary local directory.

        Parameters
        ----------
        weights_path : str
            Path prefix to the stored weights. ``"./{weights_path}/weights.keras"``
            is used as the filename.
        skip_mismatch : bool, default True
            Forwarded to ``model.load_weights``.
        by_name : bool, default True
            If ``True``, load weights by layer name.
        """
        self.model.load_weights(f'./{weights_path}/weights.keras', skip_mismatch, by_name)

    def load_model(self, run_name, log=True, load_final_state=True):
        """Load a model and optionally its final weights from MLflow.

        Parameters
        ----------
        run_name : str
            Name (or name prefix) of the MLflow run to load.
        log : bool, default True
            If ``True``, also reload configuration/model parameters from the
            run and re-log them to MLflow under a new run.
        load_final_state : bool, default True
            If ``True``, download ``final_state`` artifacts and load the
            stored weights.
        """
        run_args = mlflow.search_runs(filter_string=f"tags.mlflow.runName like '{run_name}%'" ).iloc[0]
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
        """Create a KerasTuner tuner for hyperparameter optimization.

        Parameters
        ----------
        model_type : str
            Key in ``self.model_gen_funcs`` specifying the base model
            constructor.
        model_hp : dict
            Hyperparameter search space for the model generator.
        optimizer_class : type
            Optimizer class to tune.
        optimizer_hp : dict
            Hyperparameter search space for the optimizer.
        loss_class : type
            Loss class to tune.
        loss_hp : dict
            Hyperparameter search space for the loss.
        metrics_classes : list[type]
            Metric classes to create/tune.
        metrics_hp : dict
            Hyperparameter search space for metrics.
        mlflow_log : bool
            Whether to log trials to MLflow.
        mlflow_instance : mlflow.MlflowClient or module
            MLflow instance used for logging.
        tuner_class : type
            KerasTuner tuner class to instantiate.
        tuner_args : dict
            Extra keyword arguments passed to the tuner constructor.
        """
        self.tuner = tuner_class(
            BuildHyperModel(self.model_gen_funcs[model_type], model_hp, optimizer_class, optimizer_hp, loss_class, loss_hp, metrics_classes, metrics_hp, mlflow_log, mlflow_instance),
            **tuner_args
        )

    def run_tuner(self, epochs, callbacks=[], callbacks_classes=[], callbacks_hp=[], validation_freq=1):
        """Execute the hyperparameter search using the prepared tuner.

        Parameters
        ----------
        epochs : int
            Number of training epochs per trial.
        callbacks : list[tf.keras.callbacks.Callback], optional
            Additional callbacks passed to ``tuner.search``.
        callbacks_classes : list[type], optional
            Callback classes whose hyperparameters may be tuned.
        callbacks_hp : list[dict], optional
            Hyperparameter spaces for the callbacks.
        validation_freq : int, default 1
            Validation frequency (in epochs) used during tuner search.
        """
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
        """Archive and upload local TensorBoard logs to a GCS bucket.

        The target bucket name is ``"<project-id>-tb-logs"``.

        Parameters
        ----------
        filename : str
            Base name for the zip archive and resulting blob.
        folder_path : str, default 'logs'
            Local directory containing TensorBoard log files.
        delete_after : bool, default True
            If ``True``, delete the created archive after a successful
            upload.

        Returns
        -------
        Any
            Result object from ``blob.upload_from_filename``.
        """
        shutil.make_archive(filename, 'zip', folder_path)

        bucket = storage_client.get_bucket(storage_client.project + '-tb-logs')
        result = bucket.blob(filename).upload_from_filename(filename+'.zip')

        if delete_after:
            os.remove(filename+'.zip')

        return result
    
    def download_tensorboard_logs(self, filename, save_path='', unpack_path='logs', delete_after=True):
        """Download and unpack TensorBoard logs from a GCS bucket.

        Parameters
        ----------
        filename : str
            Base name of the remote log archive (without extension).
        save_path : str, default ''
            Local directory where the zip file will be stored temporarily.
        unpack_path : str, default 'logs'
            Local directory where the archive will be extracted.
        delete_after : bool, default True
            If ``True``, delete the downloaded archive after extraction.

        Returns
        -------
        Any
            Result object from ``blob.download_to_filename``.
        """
        bucket = storage_client.get_bucket(storage_client.project + '-tb-logs')
        result = bucket.blob(filename).download_to_filename(os.path.join(save_path, filename+'.zip'))

        shutil.unpack_archive(os.path.join(save_path, filename+'.zip'), unpack_path, 'zip')

        if delete_after:
            os.remove(os.path.join(save_path, filename+'.zip'))

        return result

    ### DISPLAY RESULTS

    def print_predicted_maps(self, background_type='features', cmap='gray', alpha=0.5 ,threshold=0.5):
        """Plot a grid of predictions overlaid on input features or labels.

        Parameters
        ----------
        background_type : {'features', 'labels', 'none'}, default 'features'
            Which tensors to display as background images.
        cmap : str, default 'gray'
            Matplotlib colormap used to render the predictions.
        alpha : float, default 0.5
            Alpha value for overlaying predictions on the background.
        threshold : float, default 0.5
            Threshold used to binarize prediction probabilities.
        """

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
    """Newer training helper with more flexible dataset/model handling.

    Compared to :class:`TrainingProcessor`, this version is less coupled to
    a specific dataset generator and uses a generic model generator
    interface. It also provides more structured MLflow logging of model
    definitions and weights.

    Attributes
    ----------
    cfg : Any
        Configuration object. ``cfg.trainer_version`` is set to ``'2'``.
    mlflow : Any
        MLflow module or client used for logging.
    ds : tf.data.Dataset
        Training dataset provided via :meth:`load_dataset`.
    val_ds : Optional[tf.data.Dataset]
        Optional validation dataset.
    train_steps : int
        Number of training steps per epoch.
    val_steps : Optional[int]
        Optional number of validation steps.
    model : tf.keras.Model
        Compiled Keras model (after :meth:`compile_model`).
    initial_epoch : int
        Epoch index from which to resume training.
    run_id : str
        Active MLflow run id (when logging is enabled).
    """

    def __init__(self, cfg, mlflow_instance=None):
        self.cfg = cfg
        self.cfg.trainer_version = '2'
        self.cfg_attrs = self._get_cfg_attrs()

        self.mlflow = mlflow_instance if mlflow_instance is not None else mlflow

    def load_dataset(self, ds, train_steps, val_ds=None, val_steps=None):
        """Attach training and optional validation datasets to the trainer.

        Parameters
        ----------
        ds : tf.data.Dataset
            Training dataset.
        train_steps : int
            Number of steps per training epoch.
        val_ds : Optional[tf.data.Dataset]
            Validation dataset.
        val_steps : Optional[int]
            Number of validation steps per epoch.
        """
        self.ds = ds
        self.train_steps = train_steps
        self.val_ds = val_ds
        self.val_steps = val_steps

        self.cfg.train_samples = train_steps*self.cfg.train_batch_size
        self.cfg.val_samples = val_steps*self.cfg.val_batch_size

    def _get_cfg_attrs(self):
        """Return a list of configuration attributes for logging.

        Returns
        -------
        list[tuple[str, Any]]
            Pairs of attribute name and value for all public attributes of
            ``self.cfg``.
        """
        return [(a, getattr(self.cfg, a)) for a in dir(self.cfg) if not a.startswith('__')]
    
    def load_model_generator(self, model_generator):
        """Register a custom model generator function or callable.

        Parameters
        ----------
        model_generator : Callable
            Callable that returns a ``tf.keras.Model`` when invoked with
            ``model_args``.
        """
        self.model_generator = model_generator
        self.cfg.generator_name = model_generator.__name__
    
    def _gen_custom_model(self, model_args):
        """Instantiate a model using the registered custom generator.

        Parameters
        ----------
        model_args : dict
            Keyword arguments forwarded to ``self.model_generator``.

        Returns
        -------
        tf.keras.Model
            Model instance returned by the generator.
        """
        return self.model_generator(**model_args)

    def _log_mlflow_params(self):
        """Log model, configuration and loss parameters to MLflow.

        This method expects ``self.model_args``, ``self.cfg_attrs`` and
        ``self.loss_args`` (list of ``(name, config)``) to be populated.
        """
        
        for key in self.model_args.keys():
            self.mlflow.log_param(key, self.model_args[key])

        for key, value in self.cfg_attrs:
            self.mlflow.log_param(key, value)

        for name, loss_args in self.loss_args:
            for key, value in loss_args.items():
                self.mlflow.log_param('loss.'+name+'_'+key, value)

        #mlflow.log_param('model_type', self.model_type)

    def compile_model(self, model_args, optimizer, loss, metrics=None, weighted_metrics=None, loss_weights=None, print_summary=True, summary_kwargs={}):
        """Create and compile a model using the custom generator.

        Parameters
        ----------
        model_args : dict
            Arguments passed to the registered model generator.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer instance to use.
        loss : dict[str, tf.keras.losses.Loss]
            Dictionary mapping output names to loss instances.
        metrics : Optional[Union[list, dict]], default None
            Metrics to use during training; passed directly to
            ``model.compile``.
        weighted_metrics : Optional[list], default None
            Weighted metrics passed to ``model.compile``.
        loss_weights : Optional[list[float]], default None
            Per-output loss weights.
        print_summary : bool, default True
            If ``True``, print ``model.summary`` after compilation.
        summary_kwargs : dict, optional
            Extra keyword arguments forwarded to ``model.summary``.
        """
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
                weighted_metrics = weighted_metrics,
                jit_compile='auto'
                )

        if print_summary:
            print(self.model.summary(**summary_kwargs))

        self.initial_epoch = 0

    def _get_custom_measures(self,loss, metrics):
        """Return a dict of custom loss/metric classes for MLflow logging.

        Parameters
        ----------
        loss : dict[str, tf.keras.losses.Loss]
            Losses used by the model.
        metrics : Optional[Union[list, dict]]
            Metrics used by the model.

        Returns
        -------
        dict[str, type]
            Mapping from class name to class object for any non-keras
            components.
        """
        if metrics is None:
            metrics = {}
            
        custom_objects = {}
        for c in {**loss, **(metrics if metrics.__class__ == dict else dict([(metric.name, metric) for metric in metrics]))}.values():
            if c.__module__.split('.')[0]!='keras':
                custom_objects[c.__class__.__name__] = c
        return custom_objects

    def train_model(self, epochs, log=True, callbacks=None, export_final_state=True, export_model=False, export_model_def=True, validation_freq=1):
        """Train the compiled model with optional MLflow logging.

        Parameters
        ----------
        epochs : int
            Number of epochs to train (added to ``self.initial_epoch``).
        log : bool, default True
            If ``True``, start an MLflow run and log metrics, parameters,
            final weights, and optionally the whole model.
        callbacks : list[tf.keras.callbacks.Callback], optional
            Extra callbacks passed to ``model.fit`` in addition to the
            MLflow callback.
        export_final_state : bool, default True
            If ``True``, upload final weights as MLflow artifacts.
        export_model : bool, default False
            If ``True``, log the full model to MLflow.
        export_model_def : bool, default True
            If ``True``, upload a JSON definition of the model generator and
            compilation arguments.
        validation_freq : int, default 1
            Frequency (in epochs) at which to run validation.
        """

        if log:
            #self.mlflow.tensorflow.autolog(log_datasets=False, log_models=True, disable=False, checkpoint=False, log_every_epoch=True)
            try:
                run = self.mlflow.start_run()
            except:
                run = mlflow.active_run()
                print(f'Ending previous mlflow run {run.info.run_name}')
                mlflow.end_run()
                run = self.mlflow.start_run()

            self.run_id = run.info.run_id
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
                    'epoch': epochs+self.initial_epoch,
                    'initial_epoch': self.initial_epoch
                })
                self._log_mlflow_params()
                if export_final_state:
                    #save weights
                    self.upload_weights_to_mlflow()
                if export_model_def:
                    self.upload_model_def_to_mlflow()
                if export_model:
                    self.mlflow.tensorflow.log_model(self.model, 'model',  custom_objects=self._get_custom_measures(self.model.loss, self.model.metrics))
        
        try:
            self.mlflow.end_run()
        except:
            None

        self.initial_epoch += epochs

    def upload_weights_to_mlflow(self, run_id=None):
        """Save and upload model weights as MLflow artifacts.

        Weights are stored temporarily under ``./final_state_temp`` then
        logged to MLflow.

        Parameters
        ----------
        run_id : Optional[str]
            Specific MLflow run ID to associate the artifacts with. If not
            provided, ``self.run_id`` is used.
        """
        run_id = run_id if run_id is not None else self.run_id

        self._prepare_path('./final_state_temp')
        self.model.save_weights('./final_state_temp/final_state.weights.h5')
        self.mlflow.log_artifacts('./final_state_temp', '', run_id=run_id)

        shutil.rmtree('./final_state_temp')

    def upload_model_def_to_mlflow(self):
        """Upload a JSON description of the model definition to MLflow.

        The JSON includes model arguments, generator metadata and compiler
        function information.
        """

        self._prepare_path('./model_def_temp')
        model_def = {
            'model_args': self.model_args,
            'generator_module': self.cfg.generator_module,
            'generator_func_name': self.cfg.generator_func_name,
            'compiler_func': self.cfg.compiler_func,
            'compiler_func_args': self.cfg.compiler_func_args
        }

        with open('./model_def_temp/model_def.json', 'w') as f:
            json.dump(model_def, f)

        self.mlflow.log_artifacts('./model_def_temp', '', run_id=self.run_id)

        shutil.rmtree('./model_def_temp')


    ### LOAD MODEL

    def _load_mlflow_args(self, run_args):
        """Parse MLflow run parameters and update configuration/model args.

        This version does not reload the dataset but updates ``self.cfg``
        and internal ``loss_args`` and ``model_args`` from the run.

        Parameters
        ----------
        run_args : pandas.Series
            MLflow run information for a single run.
        """
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
        """Create a directory (and parents) if it does not yet exist.

        Parameters
        ----------
        path : str
            Directory path to create.
        """
        try:
            os.makedirs(path, exist_ok=True)
        except:
            None

    def save_temp_weights(self, weights_path, filename='final_state', use_model_name=True):
        """Save model weights to a temporary file.

        Parameters
        ----------
        weights_path : str
            Directory where weights will be stored.
        filename : str, default 'final_state'
            Base filename (without extension).
        use_model_name : bool, default True
            If ``True``, append ``self.model.name`` as a subdirectory under
            ``weights_path``.
        """

        if use_model_name:
            weights_path = os.path.join(weights_path, self.model.name)

        self._prepare_path(weights_path)
        self.model.save_weights(f'./{weights_path}/{filename}.weights.h5')

    def load_temp_weights(self, weights_path, skip_mismatch=True):
        """Load model weights from a temporary file.

        Parameters
        ----------
        weights_path : str
            Path prefix to the weights file, without the ``.weights.h5``
            extension.
        skip_mismatch : bool, default True
            Forwarded to ``model.load_weights``.
        """
        self.model.load_weights(f'./{weights_path}.weights.h5', skip_mismatch=skip_mismatch)

    def load_mlflow_weights(self, run_name, weights_path='./', skip_mismatch=True):
        """Load weights from an MLflow run using a helper downloader.

        Parameters
        ----------
        run_name : str
            Name (or name prefix) of the MLflow run.
        weights_path : str, default './'
            Unused legacy argument (kept for backward compatibility).
        skip_mismatch : bool, default True
            Forwarded to ``model.load_weights``.
        """
        run_args = self.mlflow.search_runs(filter_string=f"tags.mlflow.runName like '{run_name}%'" ).iloc[0]
        #self.initial_epoch = int(run_args['params.epochs'])

        self._download_and_load_mlflow_weights(weights_path, run_args, skip_mismatch)

    '''def _download_and_load_mlflow_weights(self, weights_path, run_args, skip_mismatch=True):
        self._prepare_path(weights_path)
        self.mlflow.artifacts.download_artifacts(run_id=run_args.run_id,artifact_path='final_state.weights.h5',
                                    dst_path=weights_path)
        self.model.load_weights(os.path.join(weights_path,'final_state.weights.h5'), skip_mismatch=skip_mismatch)'''
    
    def _download_and_load_mlflow_weights(self, run_name, skip_mismatch=True):
        """Download model weights from MLflow and load them into the model.

        Parameters
        ----------
        run_name : str
            Name (or name prefix) of the MLflow run.
        skip_mismatch : bool, default True
            Forwarded to ``model.load_weights``.
        """
        weights_path = download_mlflow_model_weights(run_name)
        self.model.load_weights(weights_path, skip_mismatch=skip_mismatch)

    def load_model(self, run_name, weights_path='./final_state/temp', load_final_state=True):
        """Load a model and optionally its final-state weights from MLflow.

        Parameters
        ----------
        run_name : str
            Name (or name prefix) of the MLflow run.
        weights_path : str, default './final_state/temp'
            Legacy argument kept for API compatibility; not used when
            ``load_final_state`` is True, since weights are downloaded via
            :func:`download_mlflow_model_weights`.
        load_final_state : bool, default True
            If ``True``, also download and load the final weights.
        """
        run_args = self.mlflow.search_runs(filter_string=f"tags.mlflow.runName like '{run_name}%'" ).iloc[0]
        self.initial_epoch = int(run_args['params.epochs']) #int(run_args['params.initial_epoch']) + 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = self.mlflow.tensorflow.load_model(os.path.join(run_args.artifact_uri, "model"))

        if load_final_state:
            self._download_and_load_mlflow_weights(weights_path, run_args, skip_mismatch=False)

        self._load_mlflow_args(run_args)