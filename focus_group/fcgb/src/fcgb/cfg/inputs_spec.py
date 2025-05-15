from fcgb.types.initial import MainSubjectModel, SubjectDetailsModel, WorkersModel, PersonaModel
from typing import TypedDict

# main subject part specs
class MainSubjectTemplateInputs(TypedDict, total=False):
    main_title: str
    initial_description: str
    customer_name: str

class MainSubjectConfig:
    initial_messages_spec = [
            {"source": "system", "template": "main_subject_system", "hidden": False},
            {"source": "ai", "template": "main_subject_hello", "hidden": False},
            {"source": "human", "template": "main_subject_human_init", "hidden": True, "as_node": "human_node"},
        ]
    button_messsage_spec = {
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

class SubjectDetailsConfig:
    initial_messages_spec = [
            {"source": "system", "template": "subject_details_system", "hidden": False},
            {"source": "ai", "template": "subject_details_hello", "hidden": False},
            {"source": "human", "template": "subject_details_human_init", "hidden": True, "as_node": "human_node"},
        ]
    button_messsage_spec = {
            'answer_format': SubjectDetailsModel,
            'template': "subject_details_button"
        }
    template_inputs_model = SubjectDetailsTemplateInputs
