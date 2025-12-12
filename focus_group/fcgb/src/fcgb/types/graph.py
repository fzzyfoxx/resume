from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated
from operator import add
from fcgb.types.utils import append_or_clear, extend_or_clear

class ContainerItemModel(BaseModel):
    name: str
    description: str

class ContainerItemsModel(BaseModel):
    items: Annotated[List[ContainerItemModel], append_or_clear]
