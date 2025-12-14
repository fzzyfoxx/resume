from fcgb.cfg.chat_inputs_spec import PhantomResearchConfig, JobHanlderConfig
from fcgb.tools.phantom_researcher import PhantomResearcherTool
from fcgb.tools.job_handler import JobHandler

# ORGANIZED RESEARCH

# phantom research tool container

class PhantomResearcherSpecTool(PhantomResearcherTool):
    def __init__(self, llm, memory=None):
        """
        Initialize the PhantomResearcherSpecTool with the given parameters.
        """
        super().__init__(
            llm=llm,
            initial_messages_spec=PhantomResearchConfig.initial_messages_spec,
            internal_messages_spec=PhantomResearchConfig.internal_messages_spec,
            memory=memory,
            init_values=PhantomResearchConfig.init_values,
            prompt_manager_spec=PhantomResearchConfig.prompt_manager_spec,
            global_inputs=PhantomResearchConfig.global_inputs
        )

        self.inputs_model = PhantomResearchConfig.template_inputs_model

# job handler

class JobHandlerSpecTool(JobHandler):
    def __init__(self, llm, tool_containers, memory=None):
        """
        Initialize the JobHandlerSpecTool with the given parameters.
        """
        super().__init__(
            llm=llm,
            tool_containers=tool_containers,
            initial_messages_spec=JobHanlderConfig.initial_messages_spec,
            internal_messages_spec=JobHanlderConfig.internal_messages_spec,
            memory=memory,
            init_values=JobHanlderConfig.init_values,
            prompt_manager_spec=JobHanlderConfig.prompt_manager_spec,
            global_inputs=JobHanlderConfig.global_inputs
        )

        self.inputs_model = JobHanlderConfig.template_inputs_model