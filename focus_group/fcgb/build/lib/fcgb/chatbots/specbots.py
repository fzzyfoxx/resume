from fcgb.chatbots.chatbot import SimpleChatBot
from fcgb.cfg.inputs_spec import MainSubjectConfig, SubjectDetailsConfig


class MainSubjectChatBot(SimpleChatBot):
    def __init__(self, llm, memory=None):
        """
        Initialize the MainSubjectChatBot with the given parameters.
        """
        super().__init__(
            llm=llm,
            initial_messages_spec=MainSubjectConfig.initial_messages_spec,
            button_messsage_spec=MainSubjectConfig.button_messsage_spec,
            memory=memory,
            button_message=True,
            prompt_manager_spec={}
        )

        self.inputs_model = MainSubjectConfig.template_inputs_model


class SubjectDetailsChatBot(SimpleChatBot):
    def __init__(self, llm, memory=None):
        """
        Initialize the SubjectDetailsChatBot with the given parameters.
        """
        super().__init__(
            llm=llm,
            initial_messages_spec=SubjectDetailsConfig.initial_messages_spec,
            button_messsage_spec=SubjectDetailsConfig.button_messsage_spec,
            memory=memory,
            button_message=True,
            prompt_manager_spec={}
        )

        self.inputs_model = SubjectDetailsConfig.template_inputs_model