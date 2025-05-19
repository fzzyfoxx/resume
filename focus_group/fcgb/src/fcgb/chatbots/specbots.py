from fcgb.chatbots.chatbot import ButtonSummaryChatBot
from fcgb.cfg.chat_inputs_spec import MainSubjectConfig, SubjectDetailsConfig


class MainSubjectChatBot(ButtonSummaryChatBot):
    def __init__(self, llm, memory=None):
        """
        Initialize the MainSubjectChatBot with the given parameters.
        """
        super().__init__(
            llm=llm,
            initial_messages_spec=MainSubjectConfig.initial_messages_spec,
            button_message_spec=MainSubjectConfig.button_message_spec,
            memory=memory,
            init_values=MainSubjectConfig.init_values,
            prompt_manager_spec={}
        )

        self.inputs_model = MainSubjectConfig.template_inputs_model


class SubjectDetailsChatBot(ButtonSummaryChatBot):
    def __init__(self, llm, memory=None):
        """
        Initialize the SubjectDetailsChatBot with the given parameters.
        """
        super().__init__(
            llm=llm,
            initial_messages_spec=SubjectDetailsConfig.initial_messages_spec,
            button_message_spec=SubjectDetailsConfig.button_message_spec,
            memory=memory,
            init_values=SubjectDetailsConfig.init_values,
            prompt_manager_spec={}
        )

        self.inputs_model = SubjectDetailsConfig.template_inputs_model