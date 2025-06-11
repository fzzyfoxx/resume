from fcgb.chatbots.chatbot import BaseChatBot
from fcgb.types.tools import ToolOutput
from langgraph.graph import MessagesState
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Annotated
from operator import add
import uuid
from langchain_core.messages import ToolMessage
from langgraph.constants import Send

class JobHandler(BaseChatBot):
    def __init__(self,
                 llm,
                 tool_containers,
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

        self.tool_containers = tool_containers

        self._set_tools()
        self.compile_graph()

    def _set_tools(self):

        tools = [container.get_tool() for container in self.tool_containers]
        self.tools_llm = self.llm.bind_tools(tools)
        self.tools = {tool.name: tool for tool in tools}

    def _set_state_class(self):

        class State(MessagesState):
            job_description: str
            turns: int = 0
            tools_threads: Annotated[List[str], add] = []
            thread_id: str = None
            output: str = None

        self.state_class = State

    def _get_system_message(self, template_inputs: Dict):
        system_prompt, _ = self._get_internal_prompt('system', template_inputs)
        system_msg = self.message_types_map['system'](system_prompt)
        return system_msg
    
    def _get_instruction_message(self, template_inputs: Dict):
        instruction_prompt, _ = self._get_internal_prompt('instruction', template_inputs)
        instruction_msg = self.message_types_map['human'](instruction_prompt)
        return instruction_msg
    
    def _get_assistant_func(self):

        def assistant(state: self.state_class): # type: ignore
            turns = state['turns'] + 1

            messages = state['messages']
            if turns==1:
                template_inputs = {'job_description': state['job_description']}
                system_msg = self._get_system_message(template_inputs)
                instruction_msg = self._get_instruction_message(template_inputs)

                messages.extend([system_msg, instruction_msg])

            if turns <= self.global_inputs['max_turns']:
                output_msg = self.tools_llm.invoke(messages)
            else:
                output_msg = self.llm.invoke(messages)

            return {'messages': output_msg, 'turns': turns}
        
        return assistant
    
    def _get_formatting_func(self):

        def formatting(state: self.state_class, config: RunnableConfig): # type: ignore
            
            thread_id = config['configurable']['thread_id']
            output = state['messages'][-1].content

            return {'thread_id': thread_id, 'output': output}
        
        return formatting
    
    def _get_tool_node_func(self):

        def tool_node_func(state: Dict): # type: ignore
            tools_outputs = []
            tools_threads = []

            tool_calls = state.get("tool_calls", [])
            for tool_call in tool_calls:
                tool = self.tools.get(tool_call['name'], None)
                if tool:
                    tool_output = tool.invoke(tool_call['args'])
                    tool_msg = ToolMessage(
                        content=tool_output['output'],
                        name=tool_call['name'],
                        tool_call_id=tool_call['id']
                    )
                    tools_outputs.append(tool_msg)
                    tools_threads.append(tool_output['thread_id'])
            return {"messages": tools_outputs, "tools_threads": tools_threads}
        
        return tool_node_func
    
    def _get_router_func(self):

        def router(state: self.state_class): # type: ignore

            last_message = state['messages'][-1]
            if len(last_message.tool_calls) > 0:
                # If there are tool calls, distribute them along tool nodes
                return [Send("tool_node", {'tool_calls': [tool_call]}) for tool_call in last_message.tool_calls]
            else:
                # If no tool calls, continue to the format node
                return 'format_node'
            
        return router
    
    def _compile_graph(self):
        
        assistant_func = self._get_assistant_func()
        formatting_func = self._get_formatting_func()
        tool_node_func = self._get_tool_node_func()
        router_func = self._get_router_func()

        workflow = StateGraph(self.state_class)
        workflow.add_node('assistant', assistant_func)
        workflow.add_node('tool_node', tool_node_func)
        workflow.add_node('format_node', formatting_func)

        workflow.add_edge(START, 'assistant')
        workflow.add_conditional_edges('assistant', router_func, ["tool_node", "format_node"])
        workflow.add_edge('tool_node', 'assistant')
        workflow.add_edge('format_node', END)

        self.graph = workflow.compile(
            checkpointer=self.memory
        )

    def run(self, job_description: str) -> ToolOutput:
        """
        Run the job handler with the given job description.
        
        Args:
            job_description (str): The description of the job to be processed.
        
        Returns:
            ToolOutput: A list of outputs from the tools used in the job processing.
        """
        thread_id = uuid.uuid4().hex
        return self.graph.invoke(
            input={
                'job_description': job_description,
                'turns': 0
                },
            config = {'configurable': {'thread_id': thread_id}},
            output_keys=['thread_id', 'output']
        )