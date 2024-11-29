import json
import os
import shutil
from importlib import import_module
import mlflow

def prepare_path(path):
    try:
        os.makedirs(path, exist_ok=True)
    except:
        None

def get_mlflow_run_id_by_name(run_name):
    return mlflow.search_runs(filter_string=f"tags.mlflow.runName like '{run_name}%'",search_all_experiments=True).iloc[0].run_id

def download_model_def(run_id, dst_path):
    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='model_def.json', dst_path=dst_path)

def download_model_weights(run_id, dst_path):
    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='final_state.weights.h5', dst_path=dst_path)

def download_mlflow_model_components(run_name, load_weights=True, dst_path='./mlflow_model_temp'):

    run_id = get_mlflow_run_id_by_name(run_name)

    prepare_path(dst_path)

    download_model_def(run_id=run_id, dst_path=dst_path)

    with open(os.path.join(dst_path, 'model_def.json')) as json_model_def:
        model_def = json.load(json_model_def)

    if load_weights:
        download_model_weights(run_id=run_id, dst_path=dst_path)

    return model_def

def delete_temp_path(dst_path='./mlflow_model_temp', files_limit=3):
    files_num = 0
    for _,_,filelist in os.walk(dst_path):
        files_num += len(filelist)
    
    if files_num<=files_limit:
        shutil.rmtree(dst_path)
    else:
        print(f'Number of files in provided directory exceeds limit.\nFiles number: {files_num}\nFiles limit: {files_limit}')

def load_mlflow_model(run_name, load_weights=True, compile=False, dst_path='./mlflow_model_temp'):

    model_def = download_mlflow_model_components(run_name=run_name, load_weights=load_weights, dst_path=dst_path)

    model_generator = getattr(import_module(model_def['generator_module']), model_def['generator_func_name'])

    model = model_generator(**model_def['model_args'])

    if load_weights:
        model.load_weights(os.path.join(dst_path, 'final_state.weights.h5'))

    if compile:
        compile_args_gen = getattr(import_module(model_def['compiler_func']), 'get_compile_args')
        compile_args = compile_args_gen(**model_def['compiler_func_args'])
        model.compile(**compile_args)

    delete_temp_path(dst_path)

    return model

def backbone_loader(
        run_name=None,
        load_mlflow_weights=None,
        backbone_args=None,
        backbone_generator=None,
        generator_func_name=None,
        weights_path=None,
        load_mode='mlflow',
        **kwargs
    ):
    """
    load_mode: 
        -'mlflow' to load model from mlflow run spec uses (run_name, load_mlflow_weights)
        -'local' to load model using generator func and locally saved weights, uses (backbone_args, backbone_modelu, generator_func_name, weights_path)
    """

    if load_mode=='mlflow':
        backbone = load_mlflow_model(run_name=run_name, load_weights=load_mlflow_weights, compile=False, dst_path='./mlflow_backbone_temp')
    elif load_mode=='local':
        generator = getattr(import_module(backbone_generator), generator_func_name)
        backbone = generator(**backbone_args)
        if weights_path is not None:
            backbone.load_weights(weights_path)
    else:
        raise ValueError(f'Wrong "load_mode" provided: {load_mode}. Choose either "mlflow" or "local"')
    
    return backbone


