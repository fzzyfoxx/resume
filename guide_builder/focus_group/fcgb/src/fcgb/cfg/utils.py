from pydantic import BaseModel
from typing import Dict

def extract_class_variables(cls):
    """Extract class-level attributes and their values, including inherited ones."""
    attributes = {}
    for base in cls.__mro__:  # Traverse the Method Resolution Order (MRO)
        attributes.update({
            key: value
            for key, value in base.__dict__.items()
            if not key.startswith("__") and not callable(value) and not isinstance(value, classmethod)
        })
    return attributes

def extract_named_class_variables(cls, vars):
    """Extract class-level attributes and their values, including inherited ones."""
    attributes = {}
    for base in cls.__mro__:  # Traverse the Method Resolution Order (MRO)
        attributes.update({
            key: value
            for key, value in base.__dict__.items()
            if not key.startswith("__") and not callable(value) and not isinstance(value, classmethod) and key in vars
        })
    return attributes

class BaseConfig:
    """Base configuration class."""

    @classmethod
    def params(cls):
        """Extract all class-level attributes and their values, including inherited ones."""
        return extract_class_variables(cls)
    
    @classmethod
    def named_params(cls, vars):
        """Extract specified class-level attributes and their values, including inherited ones."""
        return extract_named_class_variables(cls, vars)
    
def model2string(model: BaseModel) -> str:

    return '\n'.join([f'**{key}**: {value}' for key, value in model.model_dump().items()])

def dict2string(model: Dict) -> str:

    return '\n'.join([f'**{key}**: {value}' for key, value in model.items()])