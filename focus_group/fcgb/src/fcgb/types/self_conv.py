from typing import List, TypedDict, Dict
from pydantic import BaseModel, RootModel

class SelfConvModel(BaseModel):
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

class VerificationPromptsModel(BaseModel):
    verification_prompts: List[str]