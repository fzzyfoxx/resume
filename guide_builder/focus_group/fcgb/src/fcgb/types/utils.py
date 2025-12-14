from typing import List, Annotated, Union
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage

def append_or_clear(left, right):
    if right=='__clear__':
        return []
    elif right is None:
        return left
    elif left is None:
        return [right]
    elif isinstance(right, list):
        return left + right
    else:
        return left + [right]
            
def extend_or_clear(left, right):
    if right=='__clear__':
        return []
    elif right is None:
        return left
    elif left is None:
        return right
    else:
        return left + right
    
MessagesType = Annotated[List[Union[ToolMessage, AIMessage, HumanMessage, SystemMessage]], append_or_clear]