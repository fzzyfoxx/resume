from fcgb.chatbots.chatbot import BaseChatBot
from typing import Dict, Any, List, Annotated
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from fcgb.types.utils import append_or_clear
from langchain_core.runnables.config import RunnableConfig

from fcgb.types.utils import append_or_clear
from langgraph.constants import Send
from fcgb.types.self_conv import Strategyroutingstate

class SelfConversationChatBot(BaseChatBot):
    def __init__(
            self, 
            llm,
            initial_messages_spec,
            internal_messages_spec,
            memory,
            global_inputs={},
            init_values={},
            prompt_manager_spec = {}
        ):

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

        summary_type = self.internal_prompts['summary']['answer_format']

        class SelfConvState(BaseModel):
            phantom_perspective: Annotated[List[Any], append_or_clear] #MessagesState
            llm_perspective: Annotated[List[Any], append_or_clear] #MessagesState
            template_inputs: Dict[str, str]
            sc_thread_id: str | None
            parent_thread_id: str
            to_summary: bool
            turn: int
            sc_summary: summary_type | None # type: ignore

        self.state_class = SelfConvState
        self.summary_model = summary_type

    def _set_llm_func(self):
        
        def llm_call(state: self.state_class) -> Dict: # type: ignore

            response = self.llm.invoke(state.llm_perspective)
            response.name = 'llm'
            
            return {'llm_perspective': response, 'phantom_perspective': HumanMessage(response.content, name='llm')}
        
        return llm_call
    
    def _set_phantom_func(self):
        
        def phantom_call(state: self.state_class) -> Dict: # type: ignore
            turn = state.turn
            if turn >= self.global_inputs['max_turns_num']:
                outputs = {'to_summary': True}
            else:
                turn += 1
                response = self.llm.invoke(state.phantom_perspective)
                if '__end__' in response.content:
                    outputs = {'to_summary': True}
                else:
                    response.name = 'phantom'
                    outputs = {'phantom_perspective': response, 'llm_perspective': HumanMessage(response.content, name='phantom')}
        
            return outputs | {'turn': turn}
        
        return phantom_call
    
    def _set_router_func(self):

        def router_func(state: self.state_class) -> Dict: # type: ignore
            if state.to_summary:
                return 'summary_node'
            else:
                return 'llm_node'
            
        return router_func
    
    def _set_summary_func(self):

        def summary_llm_call(state: self.state_class, config: RunnableConfig) -> Dict: # type: ignore
            summary_task = HumanMessage(self.internal_prompts['summary']['prompt'].format(**state.template_inputs, **self.global_inputs))
            response = self.llm.with_structured_output(self.summary_model).invoke(state.phantom_perspective + [summary_task])

            sc_thread_id = config['configurable']['thread_id']
            return {'sc_summary': response, 'sc_thread_id': sc_thread_id, 'to_summary': False}
        
        return summary_llm_call

    
    def _compile_graph(self):

        phantom_func = self._set_phantom_func()
        llm_func = self._set_llm_func()
        router_func = self._set_router_func()
        summary_llm_call_func = self._set_summary_func()

        workflow = StateGraph(self.state_class)
        workflow.add_node('phantom_node', phantom_func)
        workflow.add_node('llm_node', llm_func)
        workflow.add_node('summary_node', summary_llm_call_func)

        workflow.add_edge(START, 'phantom_node')
        workflow.add_conditional_edges('phantom_node', router_func, ['summary_node', 'llm_node'])
        workflow.add_edge('llm_node', 'phantom_node')
        workflow.add_edge('summary_node', END)

        self.graph = workflow.compile(
            checkpointer=self.memory
        )
    
    def run(self, template_inputs: Dict, sc_thread_id: str, parent_thread_id: str):

        self.init_thread(sc_thread_id, template_inputs=template_inputs)

        config = self._get_config(sc_thread_id)

        return self.graph.invoke({'parent_thread_id': parent_thread_id, 'template_inputs': template_inputs}, config=config)
    

class StrategizedSelfConversationChatBot(BaseChatBot):
    def __init__(
            self, 
            llm,
            self_conv_bot,
            initial_messages_spec,
            internal_messages_spec,
            memory,
            global_inputs,
            init_values={},
            prompt_manager_spec = {}
        ):

        self.self_conv_bot = self_conv_bot

        super().__init__(
            llm=llm,
            initial_messages_spec=initial_messages_spec,
            internal_messages_spec=internal_messages_spec,
            memory=memory,
            init_values=init_values,
            prompt_manager_spec=prompt_manager_spec,
            global_inputs=global_inputs | self.self_conv_bot.global_inputs,
            compile=False
            )

        self.compile_graph()

    def _set_state_class(self):
        
        strategy_task_model = self.internal_prompts['strategy_task'].get('answer_format', None)
        summary_model = self.internal_prompts['summary'].get('answer_format', None)

        class StrategizedSelfConvState(BaseModel):
            template_inputs: Dict[str, str]
            ssc_thread_id: str | None
            parent_thread_id: str
            ssc_summary: summary_model | None # type: ignore
            strategies: strategy_task_model | None # type: ignore
            sc_thread_id: Annotated[List[str], append_or_clear]
            sc_summary: Annotated[List[str], append_or_clear] # type: ignore

        self.state_class = StrategizedSelfConvState

    def _set_strategy_task_func(self):

        def strategy_task_func(state: self.state_class, config: RunnableConfig) -> Dict: # type: ignore

            prompt = self.internal_prompts['strategy_task']['prompt'].format(**state.template_inputs, **self.global_inputs)
            response = self.llm.with_structured_output(self.internal_prompts['strategy_task']['answer_format']).invoke(prompt)

            return {'strategies': response, 'ssc_thread_id': config['configurable']['thread_id']}
        
        return strategy_task_func
    
    @staticmethod
    def _get_sub_thread_id(thread_id: str, i: int) -> str:
        return f'{thread_id}/self_conv{i}'
    
    def _set_router_func(self):

        def router_func(state: self.state_class) -> Dict: # type: ignore
            return [Send('self_conv_node', {
                'template_inputs': state.template_inputs | strategy,
                'sc_thread_id': self._get_sub_thread_id(state.ssc_thread_id, i),
                'parent_thread_id': state.ssc_thread_id,
                'sc_summary': None
            }) for i, strategy in enumerate(state.strategies.strategies)]
        
        return router_func
    
    def _set_conv_node_func(self):

        def conv_node_func(state: Strategyroutingstate) -> Dict: # type: ignore
            response = self.self_conv_bot.run(
                template_inputs=state['template_inputs'],
                sc_thread_id=state['sc_thread_id'],
                parent_thread_id=state['parent_thread_id']
            )
            return {'sc_summary': response['sc_summary'].answer, 'sc_thread_id': state['sc_thread_id']}
        
        return conv_node_func

    @staticmethod
    def _concat_results(results: List[str]) -> str:
        return '\n'.join([f'Result {i+1}:\n' + r for i,r in enumerate(results)])
    
    def _set_summary_func(self):

        def summary_func(state: self.state_class) -> Dict: # type: ignore
            strategized_results = self._concat_results(state.sc_summary)

            prompt = self.internal_prompts['summary']['prompt'].format(
                task=state.template_inputs['task'],
                context=state.template_inputs['context'],
                strategized_results=strategized_results,
                **self.global_inputs
            )
            response = self.llm.with_structured_output(self.internal_prompts['summary']['answer_format']).invoke(prompt)
            return {'ssc_summary': response}
        
        return summary_func
    
    def _compile_graph(self):
        
        strategy_task_func = self._set_strategy_task_func()
        router_func = self._set_router_func()
        conv_node_func = self._set_conv_node_func()
        summary_func = self._set_summary_func()

        workflow = StateGraph(self.state_class)
        workflow.add_node('strategy_task_node', strategy_task_func)
        workflow.add_node('self_conv_node', conv_node_func)
        workflow.add_node('summary_node', summary_func)

        workflow.add_edge(START, 'strategy_task_node')
        workflow.add_conditional_edges('strategy_task_node', router_func, ['self_conv_node'])
        workflow.add_edge('self_conv_node', 'summary_node')
        workflow.add_edge('summary_node', END)

        self.graph = workflow.compile(
            checkpointer=self.memory
        )

    def run(self, template_inputs: Dict, ssc_thread_id: str, parent_thread_id: str):

        self.init_thread(ssc_thread_id, template_inputs)

        config = self._get_config(ssc_thread_id)

        return self.graph.invoke({'parent_thread_id': parent_thread_id, 'template_inputs': template_inputs}, config=config)