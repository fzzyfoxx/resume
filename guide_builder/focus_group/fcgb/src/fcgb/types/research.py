from typing import List, TypedDict, Dict
from pydantic import BaseModel

class SimpleAnswerModel(BaseModel):
    answer: str

class SingleStrategyModel(TypedDict):
    strategy_description: str
    paraphrased_task: str
    paraphrased_context: str

class StrategyTaskModel(BaseModel):
    strategies: List[SingleStrategyModel]

class Strategyroutingstate(BaseModel):
    template_inputs: Dict[str, str]
    sc_thread_id: str
    parent_thread_id: str
    sc_summary: str | None

class PromptTemplatesListModel(BaseModel):
    analysis: str
    prompts: List[str]

class SingleVerificationModel(BaseModel):
    analysis: str
    recommendations: str

class SimpleTaskModel(TypedDict):
    task: str
    template_inputs: Dict[str, str]
    simple_task_response: str | None