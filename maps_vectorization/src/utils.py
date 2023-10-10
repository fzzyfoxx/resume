import importlib
import sys

def reload_script(module_name):
    importlib.reload(sys.modules[module_name])