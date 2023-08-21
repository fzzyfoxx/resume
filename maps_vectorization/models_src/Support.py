import tensorflow as tf
import keras_tuner as kt
import mlflow

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