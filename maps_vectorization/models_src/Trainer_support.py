
import os
import keras_tuner as kt
import mlflow
import tensorflow as tf
import warnings

mlflow.tensorflow.autolog(log_datasets=False, log_models=False, disable=True)

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
        

def load_mlflow_model(run_name, load_final_state=True, dst_path='.', experiment_id=None):
    run_args = mlflow.search_runs(experiment_ids=experiment_id, filter_string=f"tags.mlflow.runName like '{run_name}%'").iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mlflow.tensorflow.load_model(os.path.join(run_args.artifact_uri, "model"))

    if load_final_state:
        mlflow.artifacts.download_artifacts(run_id=run_args.run_id,artifact_path='final_state',
                                dst_path=dst_path)
        model.load_weights(os.path.join(dst_path,'final_state/variables'))

    return model

def download_mlflow_weights(run_name, experiment_id=None, dst_path='.'):
    run_args = mlflow.search_runs(experiment_ids=experiment_id, filter_string=f"tags.mlflow.runName like '{run_name}%'").iloc[0]
    mlflow.artifacts.download_artifacts(run_id=run_args.run_id,artifact_path='final_state',
                                dst_path=dst_path)