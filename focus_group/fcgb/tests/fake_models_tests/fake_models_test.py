from pydantic import BaseModel
from typing import Any, Dict, List, Tuple

class nnmodel(BaseModel):
    nn_str: str
    nn_list: List[bool]

class nmodel(BaseModel):
    n_str: str
    n_bool: bool
    n_list: List[Dict[str, bool]]
    n_dict: Dict[int, str]
    n_models_list: List[nnmodel]

class amodel(BaseModel):
    str_var: str
    int_var: int
    list_var: List[str]
    dict_var: Dict[str, Any]
    tuple_var: Tuple[str, str, str]
    tuple_list: List[Tuple[str, int]]
    bool_var: bool
    float_var: float
    none_var: None
    list_of_models: List[nmodel]