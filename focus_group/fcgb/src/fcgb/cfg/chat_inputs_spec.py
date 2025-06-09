from fcgb.types.initial import MainSubjectModel, SubjectDetailsModel, WorkersModel, RestrictionsModel
from fcgb.types.research import SimpleAnswerModel, SingleStrategyModel, StrategyTaskModel, PromptTemplatesListModel, SingleVerificationModel
from typing import TypedDict, Dict
from pydantic import BaseModel

class InternalMessageSpec(TypedDict):
    answer_format: BaseModel
    template_path: str

InternalMessagesSpec = Dict[str, InternalMessageSpec]

### BUTTON SUMMARY INITIAL MODULES

class ButtonSummaryConfig:
    init_values = {'summary': None, 'button': False}

# main subject part specs
class MainSubjectTemplateInputs(TypedDict, total=False):
    main_title: str
    initial_description: str
    customer_name: str

class MainSubjectConfig(ButtonSummaryConfig):
    initial_messages_spec = [
            {"source": "system", "template": "main_subject_system", "hidden": False},
            {"source": "ai", "template": "main_subject_hello", "hidden": False},
            {"source": "human", "template": "main_subject_human_init", "hidden": True, "as_node": "human_node"},
        ]
    internal_messages_spec = {
        'button_message': {
                'answer_format': MainSubjectModel,
                'template': "main_subject_button"
            }
    }
    template_inputs_model = MainSubjectTemplateInputs
    prompt_manager_spec = {}


#subject details part specs
class SubjectDetailsTemplateInputs(TypedDict, total=False):
    main_title: str
    initial_description: str
    customer_name: str
    main_subject: str

class SubjectDetailsConfig(ButtonSummaryConfig):
    initial_messages_spec = [
            {"source": "system", "template": "subject_details_system", "hidden": False},
            {"source": "ai", "template": "subject_details_hello", "hidden": False},
            {"source": "human", "template": "subject_details_human_init", "hidden": True, "as_node": "human_node"},
        ]
    internal_messages_spec = {
        'button_message': {
                'answer_format': SubjectDetailsModel,
                'template': "subject_details_button"
            }
    }
    template_inputs_model = SubjectDetailsTemplateInputs
    prompt_manager_spec = {}

# personas definition part specs
class PersonasTemplateInputs(TypedDict, total=False):
    main_title: str
    initial_description: str
    customer_name: str
    main_subject: str
    content_description: str
    style: str
    target_audience: str
    layout: str
    restrictions: str

class PersonasConfig(ButtonSummaryConfig):
    initial_messages_spec = [
            {"source": "system", "template": "define_personas_system", "hidden": False},
            {"source": "ai", "template": "define_personas_hello", "hidden": False}
        ]
    internal_messages_spec = {
        'button_message': {
                'answer_format': WorkersModel,
                'template': "define_personas_button"
            }
    }
    global_inputs = {
        'personas_number': 4
    }
    template_inputs_model = PersonasTemplateInputs
    prompt_manager_spec = {}

# restrictions definition part specs
class RestrictionsTemplateInputs(TypedDict, total=False):
    main_title: str
    initial_description: str
    customer_name: str
    main_subject: str
    content_description: str
    style: str
    target_audience: str
    layout: str
    restrictions: str

class RestrictionsConfig(ButtonSummaryConfig):
    initial_messages_spec = [
            {"source": "system", "template": "init_restrictions_system", "hidden": False},
            {"source": "ai", "template": "init_restrictions_hello", "hidden": False}
        ]
    internal_messages_spec = {
        'button_message': {
                'answer_format': RestrictionsModel,
                'template': "init_restrictions_button"
            }
    }
    global_inputs = {}
    template_inputs_model = RestrictionsTemplateInputs
    prompt_manager_spec = {}

### SELF-CONVERSATION MODULES

# casual self-conversation
class SelfConversationTemplateInputs(TypedDict, total=False):
    task: str
    context: str

class SelfConversationConfig:
    initial_messages_spec = [
        {"var_name": "phantom_perspective", "source": "system", "template": "self_conv_researcher_system", "hidden": False},
        {"var_name": "phantom_perspective", "source": "human", "template": "self_conv_human_init", "hidden": False},
        {"var_name": "llm_perspective", "source": "system", "template": "self_conv_expert_system", "hidden": False}
    ]
    internal_messages_spec = {
        'summary': {
                'answer_format': SimpleAnswerModel,
                'template': "self_conv_summary"
            }
    }
    template_inputs_model = SelfConversationTemplateInputs
    global_inputs = {'max_turns_num': 6}
    init_values = {'to_summary': False, 'turn': 0, 'sc_summary': None, 'sc_thread_id': None}
    prompt_manager_spec = {'version_config': {
        "self_conv_summary": "1.0",
        "self_conv_researcher_system": "1.0",
    }}


# strategized self-conversation
class SelfConversationForstrategyTemplateInputs(SelfConversationTemplateInputs, SingleStrategyModel):
    pass

class SelfConversationForstrategyConfig(SelfConversationConfig):
    template_inputs_model = SelfConversationForstrategyTemplateInputs
    prompt_manager_spec = {}

class StrategizedSelfConversationConfig:
    initial_messages_spec = []
    global_inputs = {
        'min_strategies_num': 1, 
        'max_strategies_num': 4
        }
    internal_messages_spec = {
        'strategy_task': {
                'answer_format': StrategyTaskModel,
                'template': "self_conv_strategy_list"
            },
        'summary': {
                'answer_format': SimpleAnswerModel,
                'template': "self_conv_strategy_summary"
            },
        'verification_prompts': {
                'answer_format': PromptTemplatesListModel,
                'template': "verification_prompts_gen"
            }
        }
    template_inputs_model = SelfConversationTemplateInputs
    init_values = {'ssc_thread_id': None, 'ssc_summary': None, 'strategies': None}
    prompt_manager_spec = {}

### TASK OUTPUT VERIFICATION

class ResearchVerificationTemplateInputs(TypedDict, total=False):
    task: str
    context: str
    answer: str

class ResearchVerificationConfig:
    initial_messages_spec = []
    global_inputs = {
        'min_ver_prompts': 1,
        'max_ver_prompts': 8,
    }
    internal_messages_spec = {
        'task_list': {
                'answer_format': PromptTemplatesListModel,
                'template': "verification_prompts_gen"
            },
        'task': {
                'answer_format': SingleVerificationModel,
                'template': "verification_task_system"
            },
        'answer': {
                'answer_format': SimpleAnswerModel,
                'template': "verification_output"
            }
        }
    template_inputs_model = ResearchVerificationTemplateInputs
    init_values = {'verified_answer': None, 'task_list': None}
    prompt_manager_spec = {}