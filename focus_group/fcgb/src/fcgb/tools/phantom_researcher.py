from fcgb.chatbots.chatbot import BaseChatBot
from fcgb.types.tools import ToolOutput
from langgraph.graph import MessagesState
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
import uuid


class PhantomResearcherTool(BaseChatBot):
    """
    A BaseChatBot subclass designed to mimic a tool that performs external research tasks for development and testing purposes.

    This class provides a framework for simulating research workflows, including generating system and job prompts, 
    invoking an LLM to process the research task, and returning the results in a structured format.

    Attributes:
        state_class: The state class used to define the structure of the chatbot's state.
        graph: The compiled state graph for managing the research workflow.
    """

    def __init__(self,
                 llm,
                 initial_messages_spec,
                 internal_messages_spec,
                 memory=None,
                 init_values={},
                 prompt_manager_spec={},
                 global_inputs={}):
        """
        Initialize the PhantomResearcherTool with the given parameters.

        Args:
            llm: The language model used for generating responses.
            initial_messages_spec: Specification for the initial messages.
            internal_messages_spec: Specification for internal messages.
            memory: Optional memory object for state persistence.
            init_values: Initial values for the chatbot's state.
            prompt_manager_spec: Specifications for managing prompts.
            global_inputs: Global inputs shared across the chatbot.
        """
        super().__init__(
            llm=llm,
            initial_messages_spec=initial_messages_spec,
            internal_messages_spec=internal_messages_spec,
            memory=memory,
            init_values=init_values,
            prompt_manager_spec=prompt_manager_spec,
            global_inputs=global_inputs,
            compile=False
        )
        self.compile_graph()

    def _set_state_class(self):
        """
        Define the state class for the PhantomResearcherTool, specifying the structure of the state.

        The state includes attributes for the research job, restrictions, output format, additional data, 
        thread ID, and the final output.
        """
        class State(MessagesState):
            job: str
            restrictions: str
            output_format: str
            data: str
            thread_id: str = None
            output: str = None

        self.state_class = State

    def _set_research_func(self):
        """
        Define the research function for the chatbot's state graph.

        This function generates system and job prompts, invokes the LLM to process the research task, 
        and updates the state with the results.

        Returns:
            Callable: A function that takes the chatbot's state and configuration, performs the research, 
            and returns the updated state.
        """
        def run_phantom_research(state: self.state_class, config: RunnableConfig): #type: ignore
            
            template_inputs = {
                "job": state['job'],
                "restrictions": state['restrictions'],
                "output_format": state['output_format'],
                "data": state['data']
            }

            thread_id = config['configurable']['thread_id']

            # system message
            system_prompt, _ = self._get_internal_prompt('system', template_inputs)
            system_msg = self.message_types_map['system'](system_prompt)

            # job message
            job_prompt, _ = self._get_internal_prompt('job', template_inputs)
            job_msg = self.message_types_map['human'](job_prompt)

            messages = [system_msg, job_msg]

            output = self.llm.invoke(messages)
            messages.append(output)

            return {
                'messages': messages,
                'thread_id': thread_id,
                'output': output.content
            }
        
        return run_phantom_research

    def _compile_graph(self):
        """
        Compile the state graph for the PhantomResearcherTool.

        This method sets up the workflow for the research process, including defining the research node 
        and transitions between the start, research, and end nodes.
        """
        research_func = self._set_research_func()

        workflow = StateGraph(self.state_class)
        workflow.add_node('research', research_func)

        workflow.add_edge(START, 'research')
        workflow.add_edge('research', END)

        self.graph = workflow.compile(
            checkpointer=self.memory
        )

    def run(self, job: str, restrictions: str, output_format: str, data: str) -> ToolOutput:
        """
        Perform research using the specified parameters.

        Args:
            job: A description of the research task.
            restrictions: Any limitations or constraints for the research.
            output_format: The desired format for the research output.
            data: Additional data to be used in the research.

        Returns:
            ToolOutput: The output of the research task, including the thread ID and the final result.
        """
        thread_id = uuid.uuid4().hex
        return self.graph.invoke(
            input={
                'job': job,
                'restrictions': restrictions,
                'output_format': output_format,
                'data': data
                },
            config = {'configurable': {'thread_id': thread_id}},
            output_keys=['thread_id', 'output']
        )
    
    def get_tool(self):
        """
        Create and return a tool function for performing external research.

        The tool function wraps the `run` method and provides a standardized interface for invoking 
        the research process.

        Returns:
            Callable: A tool function that takes the research parameters and returns the research output.
        """
        @tool
        def external_research(job: str, restrictions: str=None, output_format: str=None, data: str=None) -> ToolOutput:
            """
            Perform research using external sources like web pages, research papers, or Wikipedia.

            Args:
                job: A description of the research task.
                restrictions: Any limitations or constraints for the research.
                output_format: The desired format for the research output.
                data: Additional data to be used in the research.

            Returns:
                ToolOutput: The output of the research task, including the thread ID and the final result.
            """
            return self.run(job, restrictions, output_format, data)
        
        return external_research