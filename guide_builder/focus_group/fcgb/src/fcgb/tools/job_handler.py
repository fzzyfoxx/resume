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
    """
    A chatbot class for handling jobs that involve tool invocation and LLM interactions.
    This class extends the BaseChatBot and provides functionality for managing tools, 
    processing job descriptions, and generating outputs through a state graph workflow.

    Attributes:
        tool_containers: A list of tool containers that provide tools for the chatbot.
        tools_llm: A language model instance bound to the tools for tool invocation.
        tools: A dictionary mapping tool names to tool instances.
    """

    def __init__(self,
                 llm,
                 tool_containers,
                 initial_messages_spec,
                 internal_messages_spec,
                 memory=None,
                 init_values={},
                 prompt_manager_spec={},
                 global_inputs={}):
        """
        Initialize the JobHandler with the given parameters.

        Args:
            llm: The language model used for generating responses.
            tool_containers: A list of tool containers that provide tools for the chatbot.
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

        self.tool_containers = tool_containers

        self._set_tools()
        self.compile_graph()

    def _set_tools(self):
        """
        Initialize the tools for the chatbot by retrieving them from the tool containers.
        This method binds the tools to the language model for invocation.
        """
        tools = [container.get_tool() for container in self.tool_containers]
        self.tools_llm = self.llm.bind_tools(tools)
        self.tools = {tool.name: tool for tool in tools}

    def _set_state_class(self):
        """
        Define the state class for the chatbot, specifying the structure of the state.
        The state includes attributes for job description, turns, tool threads, thread ID, and output.
        """
        class State(MessagesState):
            job_description: str
            turns: int = 0
            tools_threads: Annotated[List[str], add] = []
            thread_id: str = None
            output: str = None

        self.state_class = State

    def _get_system_message(self, template_inputs: Dict):
        """
        Retrieve the system message based on the internal prompt and template inputs.

        Args:
            template_inputs: A dictionary of inputs to format the system prompt.

        Returns:
            SystemMessage: The formatted system message.
        """
        system_prompt, _ = self._get_internal_prompt('system', template_inputs)
        system_msg = self.message_types_map['system'](system_prompt)
        return system_msg
    
    def _get_instruction_message(self, template_inputs: Dict):
        """
        Retrieve the instruction message based on the internal prompt and template inputs.

        Args:
            template_inputs: A dictionary of inputs to format the instruction prompt.

        Returns:
            HumanMessage: The formatted instruction message.
        """
        instruction_prompt, _ = self._get_internal_prompt('instruction', template_inputs)
        instruction_msg = self.message_types_map['human'](instruction_prompt)
        return instruction_msg
    
    def _get_assistant_func(self):
        """
        Define the assistant function for the chatbot's state graph.
        This function processes the job description and generates responses using the LLM or tools.

        Returns:
            Callable: A function that takes the chatbot's state and returns the updated state.
        """
        def assistant(state: self.state_class): # type: ignore
            turns = state['turns'] + 1

            messages = state['messages']
            if turns == 1:
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
        """
        Define the formatting function for the chatbot's state graph.
        This function formats the final output and associates it with the thread ID.

        Returns:
            Callable: A function that takes the chatbot's state and configuration and returns the formatted output.
        """
        def formatting(state: self.state_class, config: RunnableConfig): # type: ignore
            thread_id = config['configurable']['thread_id']
            output = state['messages'][-1].content

            return {'thread_id': thread_id, 'output': output}
        
        return formatting
    
    def _get_tool_node_func(self):
        """
        Define the tool node function for the chatbot's state graph.
        This function invokes the tools specified in the tool calls and updates the state with the tool outputs.

        Returns:
            Callable: A function that processes tool calls and returns the updated state.
        """
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
        """
        Define the router function for the chatbot's state graph.
        This function determines the next node in the workflow based on the presence of tool calls.

        Returns:
            Callable: A function that routes the state to the appropriate node.
        """
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
        """
        Compile the state graph for the chatbot, defining the nodes, edges, and transitions.
        The graph includes assistant, tool, and formatting nodes, with conditional routing.
        """
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