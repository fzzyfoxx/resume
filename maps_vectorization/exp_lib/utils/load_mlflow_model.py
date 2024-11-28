import json
import os
import shutil
from importlib import import_module

def prepare_path(path):
    try:
        os.makedirs(path, exist_ok=True)
    except:
        None

def get_mlflow_run_id_by_name(run_name, mlflow):
    return mlflow.search_runs(filter_string=f"tags.mlflow.runName like '{run_name}%'",search_all_experiments=True).iloc[0].run_id

def download_model_def(run_id, dst_path, mlflow):
    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='model_def.json', dst_path=dst_path)

def download_model_weights(run_id, dst_path, mlflow):
    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='final_state.weights.h5', dst_path=dst_path)


def load_mlflow_model(run_name, mlflow, load_weights=True, compile=False, dst_path='./mlflow_model_temp'):

    run_id = get_mlflow_run_id_by_name(run_name, mlflow)

    prepare_path(dst_path)

    download_model_def(run_id=run_id, dst_path=dst_path, mlflow=mlflow)

    with open(os.path.join(dst_path, 'model_def.json')) as json_model_def:
        model_def = json.load(json_model_def)

    model_generator = getattr(import_module(model_def['generator_module']), model_def['generator_func_name'])

    model = model_generator(**model_def['model_args'])

    if load_weights:
        download_model_weights(run_id=run_id, dst_path=dst_path, mlflow=mlflow)
        model.load_weights(os.path.join(dst_path, 'final_state.weights.h5'))

    if compile:
        compile_args_gen = getattr(import_module(model_def['compiler_func']), 'get_compile_args')
        compile_args = compile_args_gen(**model_def['compiler_func_args'])
        model.compile(**compile_args)

    shutil.rmtree(dst_path)

    return model


