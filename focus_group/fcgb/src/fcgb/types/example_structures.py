from pydantic import BaseModel
from typing import Any, Dict, List, Tuple, TypedDict

class simple_model(BaseModel):
    nn_str: str
    nn_list: List[bool]

class model_with_nesting(BaseModel):
    n_str: str
    n_bool: bool
    n_list: List[Dict[str, bool]]
    n_dict: Dict[int, str]
    n_models_list: List[simple_model]

class typed_dict_model(TypedDict):
    dict_str: str
    dict_int: int
    dict_list: List[str]
    dict_dict: Dict[str, Any]
    dict_tuple: Tuple[str, str]
    dict_nested_model: simple_model
    dict_list_of_models: List[simple_model]

class model_with_double_nesting(BaseModel):
    str_var: str
    int_var: int
    list_var: List[str]
    dict_var: Dict[str, Any]
    tuple_var: Tuple[str, str, str]
    tuple_list: List[Tuple[str, int]]
    bool_var: bool
    float_var: float
    none_var: None
    list_of_models: List[model_with_nesting]
    list_of_typed_dicts: List[typed_dict_model]
