import json
import sys
from models_src.Trainer import TrainingProcessor2
from exp_lib.utils.dict_widget import display_dict
from importlib import import_module

model_def_path = sys.argv[1] 

with open(model_def_path) as json_model_def:
    model_def = json.load(json_model_def)

cfg.generator_module = model_def['generator_module'] # type: ignore
cfg.generator_func_name = model_def['generator_func_name'] # type: ignore
cfg.compiler_func = model_def['compiler_func'] # type: ignore
cfg.compiler_func_args = model_def['compiler_func_args'] # type: ignore

model_generator = getattr(import_module(cfg.generator_module), cfg.generator_func_name)  # type: ignore

compile_args_gen = getattr(import_module(cfg.compiler_func), 'get_compile_args')  # type: ignore
model_args = model_def['model_args']


if __name__=="__main__":

    trainer = TrainingProcessor2(cfg, mlflow_instance=mlflow) # type: ignore
    trainer.load_dataset(ds, train_steps, val_ds, val_steps) # type: ignore
    trainer.load_model_generator(model_generator)

    display_dict(model_args, trainer=trainer, compile_args_func=compile_args_gen, compile_args=cfg.compiler_func_args) # type: ignore
