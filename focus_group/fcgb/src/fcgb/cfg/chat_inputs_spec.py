from fcgb.types.initial import MainSubjectModel, SubjectDetailsModel, WorkersModel, PersonaModel
from fcgb.types.self_conv import SelfConvModel
from typing import TypedDict

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
    button_message_spec = {
            'answer_format': MainSubjectModel,
            'template': "main_subject_button"
        }
    template_inputs_model = MainSubjectTemplateInputs


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
    button_message_spec = {
            'answer_format': SubjectDetailsModel,
            'template': "subject_details_button"
        }
    template_inputs_model = SubjectDetailsTemplateInputs


### SELF-CONVERSATION MODULES

class SelfConversationTemplateInputs(TypedDict, total=False):
    task: str
    context: str

class SelfConversationConfig:
    initial_messages_spec = [
        {"var_name": "phantom_perspective", "source": "system", "template": "self_conv_researcher_system", "hidden": False},
        {"var_name": "phantom_perspective", "source": "human", "template": "self_conv_human_init", "hidden": False},
        {"var_name": "llm_perspective", "source": "system", "template": "self_conv_expert_system", "hidden": False}
    ]
    summary_spec = {
        'answer_format': SelfConvModel,
        'template': "self_conv_summary"
    }
    template_inputs_model = SelfConversationTemplateInputs
    init_values = {'to_summary': False, 'turn': 0, 'summary': None, 'sub_thread_id': None}