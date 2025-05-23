from fcgb.chatbots.chatbot import ButtonSummaryChatBot
from fcgb.chatbots.selfconv import SelfConversationChatBot, StrategizedSelfConversationChatBot
from fcgb.cfg.chat_inputs_spec import MainSubjectConfig, SubjectDetailsConfig
from fcgb.cfg.chat_inputs_spec import SelfConversationConfig, SelfConversationForstrategyConfig, StrategizedSelfConversationConfig


class MainSubjectSpecBot(ButtonSummaryChatBot):
    def __init__(self, llm, memory=None):
        """
        Initialize the MainSubjectChatBot with the given parameters.
        """
        super().__init__(
            llm=llm,
            initial_messages_spec=MainSubjectConfig.initial_messages_spec,
            internal_messages_spec=MainSubjectConfig.internal_messages_spec,
            memory=memory,
            init_values=MainSubjectConfig.init_values,
            prompt_manager_spec=MainSubjectConfig.prompt_manager_spec
        )

        self.inputs_model = MainSubjectConfig.template_inputs_model


class SubjectDetailsSpecBot(ButtonSummaryChatBot):
    def __init__(self, llm, memory=None):
        """
        Initialize the SubjectDetailsChatBot with the given parameters.
        """
        super().__init__(
            llm=llm,
            initial_messages_spec=SubjectDetailsConfig.initial_messages_spec,
            internal_messages_spec=SubjectDetailsConfig.internal_messages_spec,
            memory=memory,
            init_values=SubjectDetailsConfig.init_values,
            prompt_manager_spec=SubjectDetailsConfig.prompt_manager_spec
        )

        self.inputs_model = SubjectDetailsConfig.template_inputs_model

class CasualSelfConvSpecBot(SelfConversationChatBot):
    def __init__(self, llm, memory=None):
        """
        Initialize the CasualSelfConvChatBot with the given parameters.
        """
        super().__init__(
            llm=llm,
            initial_messages_spec=SelfConversationConfig.initial_messages_spec,
            internal_messages_spec=SelfConversationConfig.internal_messages_spec,
            memory=memory,
            global_inputs=SelfConversationConfig.global_inputs,
            init_values=SelfConversationConfig.init_values,
            prompt_manager_spec=SelfConversationConfig.prompt_manager_spec
        )

        self.inputs_model = SelfConversationConfig.template_inputs_model

class SelfConvForStrategySpecBot(SelfConversationChatBot):
    def __init__(self, llm, memory=None):
        """
        Initialize the SelfConvForStrategyChatBot with the given parameters.
        """
        super().__init__(
            llm=llm,
            initial_messages_spec=SelfConversationForstrategyConfig.initial_messages_spec,
            internal_messages_spec=SelfConversationForstrategyConfig.internal_messages_spec,
            memory=memory,
            global_inputs=SelfConversationForstrategyConfig.global_inputs,
            init_values=SelfConversationForstrategyConfig.init_values,
            prompt_manager_spec=SelfConversationForstrategyConfig.prompt_manager_spec
        )

        self.inputs_model = SelfConversationForstrategyConfig.template_inputs_model

class StrategizedSelfResearchSpecBot(StrategizedSelfConversationChatBot):
    def __init__(self, llm, self_conv_bot, memory=None):
        """
        Initialize the StrategizedSelfConvChatBot with the given parameters.
        """
        super().__init__(
            llm=llm,
            self_conv_bot=self_conv_bot,
            initial_messages_spec=StrategizedSelfConversationConfig.initial_messages_spec,
            internal_messages_spec=StrategizedSelfConversationConfig.internal_messages_spec,
            memory=memory,
            global_inputs=StrategizedSelfConversationConfig.global_inputs,
            init_values=StrategizedSelfConversationConfig.init_values,
            prompt_manager_spec=StrategizedSelfConversationConfig.prompt_manager_spec
        )

        self.inputs_model = StrategizedSelfConversationConfig.template_inputs_model