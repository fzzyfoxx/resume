from fcgb.chatbots.chatbot import BaseChatBot
from fcgb.types.tools import ToolOutput
from langgraph.graph import MessagesState
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
import uuid


class PhantomResearcherTool(BaseChatBot):
    def __init__(self,
                 llm,
                 initial_messages_spec,
                 internal_messages_spec,
                 memory=None,
                 init_values={},
                 prompt_manager_spec={},
                 global_inputs={}):
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

        class State(MessagesState):
            job: str
            restrictions: str
            output_format: str
            data: str
            thread_id: str = None
            output: str = None

        self.state_class = State

    def _set_research_func(self):

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
        Performs research using external sources like web pages, research papers, wikipedia.
        Args:
            job: description of the research task
            restrictions: any limitations or constraints for the research
            output_format: the desired format for the research output
            data: additional data to be used in the research
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

        @tool
        def external_research(job: str, restrictions: str, output_format: str, data: str) -> ToolOutput:
            """
            Performs research using external sources like web pages, research papers, wikipedia.
            Args:
                job: description of the research task
                restrictions: any limitations or constraints for the research
                output_format: the desired format for the research output
                data: additional data to be used in the research
            """
            return self.run(job, restrictions, output_format, data)
        
        return external_research