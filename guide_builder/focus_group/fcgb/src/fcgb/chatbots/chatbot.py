from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import Image, display
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel
from fcgb.utils import load_txt
from fcgb.prompt_manager import PromptManager
import nest_asyncio

class ButtonMessageSpec(TypedDict):
    answer_format: BaseModel
    template_path: str


class BaseChatBot:
    """
    Base class for chatbots using LangGraph StateGraph.
    This class provides the foundational structure for creating chatbots that can manage conversations
    through a state graph, utilizing prompts and memory for state persistence.
    """
    def __init__(self,
                 llm,
                 initial_messages_spec,
                 internal_messages_spec,
                 memory=None,
                 init_values={},
                 prompt_manager_spec={},
                 global_inputs={},
                 compile=True
                 ):
        """
        Initialize the BaseChatBot with the given parameters.
        Args:
            llm: The language model class to be used for generating responses.
            initial_messages_spec: A list of dictionaries defining the initial messages for the chatbot.
            internal_messages_spec: A dictionary defining the internal message specifications.
            memory: An optional memory saver instance for state persistence. If None, a default MemorySaver is used.
            init_values: A dictionary of initial values to set in the chatbot's state.
            prompt_manager_spec: A dictionary specifying the configuration for the PromptManager.
            global_inputs: A dictionary of global input variables to be used in prompt templates.
            compile: A boolean indicating whether to compile the state graph upon initialization.
        """
        
        self.llm = llm
        self.initial_messages_spec = initial_messages_spec
        self.internal_messages_spec = internal_messages_spec

        self.memory = memory if memory else MemorySaver()

        self.init_values = init_values
        self.global_inputs = global_inputs

        self.prompt_manager = PromptManager(**prompt_manager_spec)

        self.message_types_map = {
            'system': SystemMessage,
            'human': HumanMessage,
            'ai': AIMessage,
            'remove': RemoveMessage
        }

        self.compile = compile


        if self.compile:
            self.compile_graph()

    def compile_graph(self):
        """
        Compile the state graph for the chatbot.
        """
        self._set_prompts()
        self._set_state_class()
        self._compile_graph()

    def _set_prompts(self):
        """
        Set the internal prompts for the chatbot based on the internal message specifications.
        """
        self.internal_prompts = {}
        if self.internal_messages_spec:
            for prompt_key, prompt_spec in self.internal_messages_spec.items():
                self.internal_prompts[prompt_key] = {
                    'prompt': self.prompt_manager.get_prompt(prompt_spec['template']),
                    'answer_format': prompt_spec['answer_format']
                }

    def _set_state_class(self):
        """
        Set the state class for the graph. This is a placeholder method and should be
        overridden in subclasses to provide a specific state class.
        """

        class State(MessagesState):
            pass

        self.state_class = State

    def _get_internal_prompt(self, name: str, template_inputs: Dict):
        """
        Retrieve and format an internal prompt based on the provided name and template inputs.

        Args:
            name: The name of the internal prompt to retrieve.
            template_inputs: A dictionary of inputs to format the prompt template.
        Returns:
            A tuple containing the formatted prompt string and its associated answer format (if any).
        """
        prompt = self.internal_prompts[name]['prompt'].format(**template_inputs, **self.global_inputs)
        answer_format = self.internal_prompts[name].get('answer_format', None)

        return prompt, answer_format
    
    def _get_internal_message(self, name: str, template_inputs: Dict):
        """
        Retrieve an internal message object based on the provided name and template inputs.

        Args:
            name: The name of the internal message to retrieve.
            template_inputs: A dictionary of inputs to format the prompt template.
        Returns:
            A tuple containing the message object and its associated answer format (if any).
        """
        prompt, answer_format = self._get_internal_prompt(name, template_inputs)
        role = self.internal_messages_spec[name]['role'] if 'role' in self.internal_messages_spec[name] else 'human'

        msg = self.message_types_map[role](prompt, name=name)

        return msg, answer_format

    def _invoke_internal_msg(self, name, template_inputs):
        """
        Invoke the language model with a specific internal message.

        Args:
            name: The name of the internal message to invoke.
            template_inputs: A dictionary of inputs to format the prompt template.
        Returns:
            The response from the language model.
        """
        prompt, answer_format = self._get_internal_prompt(name, template_inputs)
        
        if answer_format:
            return self.llm.with_structured_output(answer_format).invoke(prompt)
        return self.llm.invoke(prompt)

    def _apply_initial_message(self, config, template_inputs, source, template, var_name='messages', hidden=False, version=None, as_node=None):
        """
        Apply a message to the graph state.
        """
        msg_content = self.prompt_manager.get_prompt(template, version).format(**template_inputs, **self.global_inputs)
        msg = self.message_types_map[source](msg_content, name="hidden" if hidden else "basic")

        self.graph.update_state(config=config, values={var_name: msg}, as_node=as_node)

        if as_node:
            # If input is simulating a node, run graph until next interruption (designed for human input)
            self.graph.invoke(input=None, config=config)

    def init_thread(self, thread_id: str, template_inputs: Dict[str, Any] = {}):
        """
        Initialize a new thread with system message template inputs.

        Args:
            thread_id: A string representing the thread ID for the current conversation.
            template_inputs: A dictionary containing variables to be passed to the chatbot prompts templates.
        """
        config = self._get_config(thread_id)

        memory_state = self.memory.get(config)
        # initial messages are only applied if there is no state in memory
        if not memory_state:
            # set default button value
            self.graph.update_state(config=config, values=self.init_values)

            # add messages in an order provided in initial_messages_spec
            for spec in self.initial_messages_spec:
                self._apply_initial_message(config, template_inputs, **spec)

    def _get_state(self, thread_id):
        """
        Get the current state of the conversation for a given thread ID.
        
        Args:
            thread_id: A string representing the thread ID for the current conversation.
        """
        return self.graph.get_state({'configurable': {'thread_id': thread_id}})
    
    def _set_llm_func(self):
        """
        Set the LLM function for the graph. This function invokes the language model with the current messages in the state.

        Returns:
            A callable function that takes the graph state as input and returns the LLM response.
        """
        def llm_call(state: self.state_class) -> Dict: # type: ignore
            """
            A callable function that takes the graph state as input and returns the LLM response.

            Args:
                state: The current state of the graph.
            Returns:
                A dictionary containing the LLM response message.
            """
            response = self.llm.invoke(state['messages'])
            response.name = 'llm'
            return {'messages': response}
        return llm_call
    
    def _set_human_func(self):
        """
        Set the human function for the graph. This is a placeholder method and should be
        overridden in subclasses to provide specific human input handling.

        Returns:
            A callable function that takes the graph state as input.
        """
        def human_input(state: self.state_class): # type: ignore
            pass

        return human_input
    
    def _compile_graph(self):
        """
        Compile the state graph for the chatbot. This method sets up the nodes and edges of the graph,
        defining the flow of conversation between human input and LLM responses.
        """
        human_func = self._set_human_func()
        llm_func = self._set_llm_func()

        workflow = StateGraph(self.state_class)
        workflow.add_node('human_node', human_func)
        workflow.add_node('llm_node', llm_func)

        workflow.add_edge(START, 'human_node')
        workflow.add_edge('human_node', 'llm_node')
        workflow.add_edge('llm_node', END)

        self.graph = workflow.compile(
            checkpointer=self.memory,
            interrupt_before=['human_node']
        )
    
    def display_graph(self):
        """
        Display the compiled state graph using Mermaid visualization.
        """
        nest_asyncio.apply()
        display(Image(self.graph.get_graph(xray=1).draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER), height=200, width=200))

    def _get_config(self, thread_id: str):
        """
        Get the configuration dictionary for a given thread ID.

        Args:
            thread_id: A string representing the thread ID for the current conversation.
        Returns:
            A dictionary containing the configuration for the given thread ID.
        """
        return {'configurable': {'thread_id': thread_id}}
    
    @staticmethod
    def _filter_stream_output(output):
        """
        Filter the stream output to extract relevant message information.

        Args:
            output: The output stream from the graph execution.
        Returns:
            A list of tuples containing the type and content of each message.
        """
        return [(x['messages'].type, x['messages'].content) for x in [[v for v in elem.values()][0] for elem in output] if isinstance(x, dict)]
    
    def _message_handler(self, message: dict):
        """
        Handle incoming messages and prepare them for graph state update.
        Args:
            message: A dictionary containing the type of message and its value.
        Returns:
            A dictionary of values to update the graph state.
        """
        if message['type'] == 'message':
            # add pure human input
            human_msg = HumanMessage(message['value'], name='response')
            values={'messages': human_msg}
        else:
            raise ValueError(f"Unknown message type: {message['type']}")
        
        return values
    
    def response(self, message: dict, thread_id: str):
        """
        Process the response from the human input and update the graph state accordingly.

        Args:
            message: A dictionary containing the type of message and its value.
            thread_id: A string representing the thread ID for the current conversation.
        Returns:
            A list of tuples containing the type and content of each message in the response.
        """
        config = self._get_config(thread_id)

        values = self._message_handler(message)
        
        # update graph state with human message for thread_id
        self.graph.update_state(config=config, values=values, as_node='human_node')
        
        # run graph loop until next interruption
        output = self.graph.stream(input=None, config=config, stream_mode='updates', output_keys=['messages'])

        # catch graph loop outputs
        filtered_output = self._filter_stream_output(output)
        return filtered_output
    
    def get_state(self, thread_id: str):
        """
        Get the current state of the conversation for a given thread ID.

        Args:
            thread_id: A string representing the thread ID for the current conversation.
        Returns:
            A dictionary containing the current state of the conversation.
        """
        config = self._get_config(thread_id)
        state = self.memory.get(config)
        if state:
            return state['channel_values']
        return None
    
    def get_messages(self, thread_id: str, field_name: str = 'messages'):
        """
        Get all messages for a given thread ID.

        Args:
            thread_id: A string representing the thread ID for the current conversation.
        Returns:
            A list of tuples containing the type, content and name of each message.
        """
        config = self._get_config(thread_id)
        state = self.memory.get(config)
        if state:
            return [(m.type, m.content, m.name) for m in state['channel_values'][field_name]]
        return None
    
    def get_state_field(self, thread_id: str, field_name: str):
        """
        Get the State field for a given thread ID and field_name.

        Args:
            thread_id: A string representing the thread ID for the current conversation.
            field_name: A string representing the name of the field to retrieve.
        Returns:
            A string containing the requested field content of the conversation.
        """
        state = self.get_state(thread_id)
        if state:
            return state.get(field_name, None)
        return None

class ButtonSummaryChatBot(BaseChatBot):
    """
    A chatbot class that extends the BaseChatBot to include functionality for handling
    button-based interactions and generating summaries using a state graph workflow.
    """

    def __init__(self,
                 llm,
                 initial_messages_spec,
                 internal_messages_spec,
                 memory=None,
                 global_inputs={},
                 init_values={},
                 prompt_manager_spec={}
                 ):
        """
        Initialize the ButtonSummaryChatBot with the given parameters.

        Args:
            llm: The language model used for generating responses.
            initial_messages_spec: Specification for the initial messages.
            internal_messages_spec: Specification for internal messages.
            memory: Optional memory object for state persistence.
            global_inputs: Global inputs shared across the chatbot.
            init_values: Initial values for the chatbot's state.
            prompt_manager_spec: Specifications for managing prompts.
        """
        super().__init__(
            llm=llm,
            initial_messages_spec=initial_messages_spec,
            internal_messages_spec=internal_messages_spec,
            memory=memory,
            global_inputs=global_inputs,
            prompt_manager_spec=prompt_manager_spec,
            init_values=init_values,
            compile=True
        )

    def _set_state_class(self): 
        """
        Set the state class for the chatbot, defining the structure of the state
        with attributes for button and summary. The summary type is dynamically
        determined from the internal prompts.
        """
        summary_type = self.internal_prompts['button_message']['answer_format']

        class State(MessagesState):
            button: bool
            summary: summary_type # type: ignore

        self.state_class = State

    def _set_button_prompt(self, template_inputs):
        """
        Set the button prompt for the chatbot's graph state based on the internal
        prompts and the provided template inputs.

        Args:
            template_inputs (dict): A dictionary of inputs to format the button prompt.
        """
        if self.internal_prompts['button_message']:
            self.button_prompt = self.internal_prompts['button_message']['prompt'].format(**template_inputs)
            self.button_model = self.internal_prompts['button_message']['answer_format']

    def init_thread(self, thread_id: str, template_inputs: Dict[str, Any] = {}):
        """
        Initialize a new thread for the chatbot and set the button prompt.

        Args:
            thread_id (str): The unique identifier for the thread.
            template_inputs (dict, optional): Template inputs for initializing the thread.
        """
        super().init_thread(thread_id, template_inputs)
        self._set_button_prompt(template_inputs)

    def _set_summary_func(self):
        """
        Define the summary function for the chatbot's state graph. This function
        uses the language model to generate a summary based on the current state.

        Returns:
            Callable: A function that takes the chatbot's state and returns a dictionary
            with the summary and button flag.
        """
        def summary_llm_call(state: self.state_class) -> Dict: # type: ignore
            button_model = self.internal_prompts['button_message']['answer_format']
            response = self.llm.with_structured_output(button_model).invoke(state['messages'])
            return {'summary': response, 'button': False}
        return summary_llm_call
    
    def _set_button_cond_func(self) -> str:
        """
        Define the conditional function for transitioning between nodes in the state graph
        based on the button flag in the chatbot's state.

        Returns:
            Callable: A function that determines the next node based on the button flag.
        """
        def button_cond_func(state: self.state_class): # type: ignore
            if state['button']:
                return 'summary_node'
            return 'llm_node'
        
        return button_cond_func
    
    def _compile_graph(self):
        """
        Compile the state graph for the chatbot, defining the nodes, edges, and
        conditional transitions. The graph includes human, LLM, and summary nodes.
        """
        human_func = self._set_human_func()
        llm_func = self._set_llm_func()
        summary_func = self._set_summary_func()
        button_cond_func = self._set_button_cond_func()

        workflow = StateGraph(self.state_class)
        workflow.add_node('human_node', human_func)
        workflow.add_node('llm_node', llm_func)
        workflow.add_node('summary_node', summary_func)

        workflow.add_edge(START, 'human_node')
        workflow.add_conditional_edges('human_node', button_cond_func, ['summary_node', 'llm_node'])
        workflow.add_edge('llm_node', 'human_node')
        workflow.add_edge('summary_node', END)

        self.graph = workflow.compile(
            checkpointer=self.memory,
            interrupt_before=['human_node']
        )

    def _message_handler(self, message: dict):
        """
        Handle incoming messages and update the chatbot's state accordingly.

        Args:
            message (dict): A dictionary representing the incoming message. It should
            contain a 'type' key (e.g., 'message' or 'button') and a 'value' key.
        Returns:
            dict: A dictionary of values to update the chatbot's state.
        """
        if message['type'] == 'message':
            # add pure human input
            human_msg = HumanMessage(message['value'], name='response')
            values={'messages': human_msg}
        elif message['type'] == 'button':
            # update button flag and add passed button message as human input
            values = {'button': True}
            button_msg = message['value'] if message['value'] else self.button_prompt
            values['messages'] = HumanMessage(button_msg, name='hidden')
        else:
            raise ValueError(f"Unknown message type: {message['type']}")
        
        return values