import json
import sys
import os
from models_src.Trainer import TrainingProcessor2
from exp_lib.utils.dict_widget import display_dict
from exp_lib.utils.load_mlflow_model import download_mlflow_model_components, delete_temp_path, get_mlflow_run_id_by_name
from importlib import import_module
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--run_name", default='', type=str, help="If provided, model definition is loaded from mlflow run")
parser.add_argument("--load_weights", default=0, type=int, help="If >0 then mlflow model weights are downloaded and load to model")
parser.add_argument("--summary", default=1, type=int, help="If 1 then print compiled model summary")
parser.add_argument("--trainer_name", default='trainer', type=str, help="Name of the trainer class")
parser.add_argument("--auto_accept", default=1, type=int, help="Automatically accept parameters and compile model")

kwargs, args = parser.parse_known_args()#vars(parser.parse_args())
kwargs = vars(kwargs)
run_name = kwargs['run_name']
load_weights = bool(kwargs['load_weights'])
print_summary = bool(kwargs['summary'])
cache_path = '../model_cache'
trainer_name = kwargs['trainer_name']
auto_accept = bool(kwargs['auto_accept'])

if run_name!='':
    model_def = download_mlflow_model_components(run_name=run_name, load_weights=load_weights, dst_path=cache_path)
else:

    model_def_path = args[0]

    with open(model_def_path) as json_model_def:
        model_def = json.load(json_model_def)

try:
    cfg # type: ignore
except:
    print('cfg class initialized')
    class cfg:
        pass

cfg.generator_module = model_def['generator_module']
cfg.generator_func_name = model_def['generator_func_name']
cfg.compiler_func = model_def['compiler_func']
cfg.compiler_func_args = model_def['compiler_func_args']

model_generator = getattr(import_module(cfg.generator_module), cfg.generator_func_name)

compile_args_gen = getattr(import_module(cfg.compiler_func), 'get_compile_args')
model_args = model_def['model_args']


if __name__=="__main__":

    globals()[trainer_name] = TrainingProcessor2(cfg, mlflow_instance=mlflow) # type: ignore
    globals()[trainer_name].load_dataset(ds, train_steps, val_ds, val_steps) # type: ignore
    globals()[trainer_name].load_model_generator(model_generator)

    if load_weights:
        globals()[trainer_name].compile_model(
                    model_args = model_args, 
                    print_summary = print_summary,
                    summary_kwargs = {'expand_nested': False, 'line_length': 100},
                    **compile_args_gen(**cfg.compiler_func_args)
                )
        globals()[trainer_name].model.load_weights(os.path.join(cache_path, run_name, 'final_state.weights.h5'))
        #delete_temp_path(temp_path)
    
    if run_name!='':
        globals()[trainer_name].run_id = get_mlflow_run_id_by_name(run_name)
        
    else:
        display_dict(model_args, trainer=globals()[trainer_name], compile_args_func=compile_args_gen, compile_args=cfg.compiler_func_args, auto_accept=auto_accept, print_summary=print_summary) # type: ignore
