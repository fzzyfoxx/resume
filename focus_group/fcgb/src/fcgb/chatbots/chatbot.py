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
        
        self._set_prompts()
        self._set_state_class()
        self._compile_graph()

    def _set_prompts(self):

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
        return self.graph.get_state({'configurable': {'thread_id': thread_id}})
    
    def _set_llm_func(self):
        
        def llm_call(state: self.state_class) -> Dict: # type: ignore
            response = self.llm.invoke(state['messages'])
            response.name = 'llm'
            return {'messages': response}
        return llm_call
    
    def _set_human_func(self):
        def human_input(state: self.state_class): # type: ignore
            pass

        return human_input
    
    def _compile_graph(self):

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
        nest_asyncio.apply()
        display(Image(self.graph.get_graph(xray=1).draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER), height=200, width=200))

    def _get_config(self, thread_id: str):
        return {'configurable': {'thread_id': thread_id}}
    
    @staticmethod
    def _filter_stream_output(output):
        return [(x['messages'].type, x['messages'].content) for x in [[v for v in elem.values()][0] for elem in output] if isinstance(x, dict)]
    
    def _message_handler(self, message: dict):

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

        Inputs:
        - message: A dictionary containing the type of message and its value.
        - thread_id: A string representing the thread ID for the current conversation.
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

        Inputs:
        - thread_id: A string representing the thread ID for the current conversation.

        Outputs:
        - A dictionary containing the current state of the conversation.
        """
        config = self._get_config(thread_id)
        state = self.memory.get(config)
        if state:
            return state['channel_values']
        return None
    
    def get_messages(self, thread_id: str, field_name: str = 'messages'):
        """
        Get all messages for a given thread ID.

        Inputs:
        - thread_id: A string representing the thread ID for the current conversation.

        Outputs:
        - A list of tuples containing the type, content and name of each message.
        """
        config = self._get_config(thread_id)
        state = self.memory.get(config)
        if state:
            return [(m.type, m.content, m.name) for m in state['channel_values'][field_name]]
        return None
    
    def get_state_field(self, thread_id: str, field_name: str):
        """
        Get the State field for a given thread ID and field_name.

        Inputs:
        - thread_id: A string representing the thread ID for the current conversation.
        - field_name: A string representing the name of the field to retrieve.

        Outputs:
        - A string containing the requested field content of the conversation.
        """
        state = self.get_state(thread_id)
        if state:
            return state.get(field_name, None)
        return None

class ButtonSummaryChatBot(BaseChatBot):
    def __init__(self,
                 llm,
                 initial_messages_spec,
                 internal_messages_spec,
                 memory=None,
                 init_values={},
                 prompt_manager_spec={}
                 ):

        super().__init__(
            llm=llm,
            initial_messages_spec=initial_messages_spec,
            internal_messages_spec=internal_messages_spec,
            memory=memory,
            prompt_manager_spec=prompt_manager_spec,
            init_values=init_values,
            compile=True
        )

    def _set_state_class(self): 
        
        summary_type = self.internal_prompts['button_message']['answer_format']

        class State(MessagesState):
            button: bool
            summary: summary_type # type: ignore

        self.state_class = State[summary_type]

    def _set_button_prompt(self, template_inputs):
        """
        Set the button prompt for the graph state.
        """
        if self.internal_prompts['button_message']:
            self.button_prompt = self.internal_prompts['button_message']['prompt'].format(**template_inputs)
            self.button_model = self.internal_prompts['button_message']['answer_format']

    def init_thread(self, thread_id: str, template_inputs: Dict[str, Any] = {}):
        super().init_thread(thread_id, template_inputs)

        self._set_button_prompt(template_inputs)

    def _set_summary_func(self):
        def summary_llm_call(state: self.state_class) -> Dict: # type: ignore
            button_model = self.internal_prompts['button_message']['answer_format']
            response = self.llm.with_structured_output(button_model).invoke(state['messages'])
            return {'summary': response, 'button': False}
        return summary_llm_call
    
    def _set_button_cond_func(self) -> str:
        def button_cond_func(state: self.state_class): # type: ignore
            if state['button']:
                return 'summary_node'
            return 'llm_node'
        
        return button_cond_func
    
    def _compile_graph(self):

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

        if message['type'] == 'message':
            # add pure human input
            human_msg = HumanMessage(message['value'], name='response')
            values={'messages': human_msg}
        elif message['type'] == 'button':
            # update button flag and add passed butten message as human input
            values = {'button': True}
            button_msg = message['value'] if message['value'] else self.button_prompt
            values['messages'] = HumanMessage(button_msg, name='hidden')
        else:
            raise ValueError(f"Unknown message type: {message['type']}")
        
        return values