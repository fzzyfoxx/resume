import json
import random
import importlib
import sys

def save_to_json(input_string, filename):
    with open(filename, 'w') as json_file:
        json.dump(input_string, json_file, ensure_ascii=False, indent=4)

def load_from_json(filename):
    with open(filename, 'r') as json_file:
        return json.load(json_file)
    
def load_txt(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def reload_script(module_name):
    importlib.reload(sys.modules[module_name])