from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import List, Dict, TypedDict, Annotated
from pydantic import BaseModel
import json
from operator import add
from fcgb.prompt_manager import PromptManager
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm
import asyncio
from rag_eval.utils import create_path_if_not_exists, sort_dicts_by_id

class ChunkEvalModel(TypedDict):
    idx: int
    text: str
    labels: List[bool]

class ChunkEvalLabel(BaseModel):
    query_id: int
    evaluation: bool

class ChunkEvalAnswer(BaseModel):
    answer: List[ChunkEvalLabel]

class ChunkEvalState(BaseModel):
    metadata_file: str
    queries: List[str] = []
    doc_file: str = ''
    title: str = ''
    summary: str = ''
    context: str = ''
    chunks: List[str] = []
    chunks_num: int = 0
    chunk_idx: int = 0
    chunks_eval: Annotated[List[ChunkEvalModel], add] = []


class ChunkEvalGraph:
    def __init__(
            self,
            llm,
            prompts_config: Dict,
            docs_metadata_path: str,
            saving_path: str,
            memory=None,
            chunk_size: int = 600,
            chunk_overlap: int = 0,
            max_queries: int = 5,
            context_agg_interval: int = 5,
            prompt_manager_spec: Dict = {}
    ):
        """
        Initializes the ChunkEvalGraph for evaluating chunks of text based on a query.
        
        Args:
            llm: The language model to use for evaluating chunks.
            prompts_config (Dict): Configuration for prompts.
                Configuration should include:
                'path': main path where prompts are stored,
                'chunk_eval_system': title of the system prompt for chunk evaluation.
                'chunk_eval_task': title of the prompt for chunk evaluation task.
                'doc_context_system': title of the prompt for document's current context system message.
                'doc_context_update': title of the prompt for updating document's current context.
                'doc_context_aggregation': title of the prompt for aggregating document's context.
            docs_metadata_path (str): Path to save metadata of downloaded documents.
            saving_path (str): Path to save evaluation results.
            memory: Optional memory for the graph.
            chunk_size (int): Size of each chunk to evaluate.
            chunk_overlap (int): Overlap between chunks.
            max_queries (int): Maximum number of queries to evaluate per document.
            context_agg_interval (int): Interval for aggregating context.
            prompt_manager_spec (Dict): Specification for the prompt manager, especially prompts versions.
        """
        self.llm = llm
        self.prompts_config = prompts_config
        self.docs_metadata_path = docs_metadata_path
        self.saving_path = saving_path
        self.memory = memory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_queries = max_queries
        self.context_agg_interval = context_agg_interval
        self.prompt_manager_spec = prompt_manager_spec

        create_path_if_not_exists(self.saving_path)

        self._set_text_splitter()
        self._set_promps()
        self.build_graph()
    
    def _set_promps(self):
        """
        Initializes the prompt manager and retrieves the prompts for chunk evaluation.
        """
        prompt_manager = PromptManager(
            version_config=self.prompt_manager_spec,
            path=self.prompts_config['path']
        )

        self.prompts = prompt_manager.get_prompts([self.prompts_config[name] for name in ['chunk_eval_system', 'chunk_eval_task', 'doc_context_system', 'doc_context_update', 'doc_context_aggregation']])

    def _set_text_splitter(self):
        """
        Initializes the text splitter for chunking documents.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

    def chunking_node(self, state: ChunkEvalState):

        with open(os.path.join(self.docs_metadata_path, state.metadata_file), 'r') as f:
            metadata = json.load(f)

        queries = metadata['queries'][:self.max_queries]
        doc_file = metadata['path']
        try:
            loader = PyPDFLoader(doc_file, mode='single', extraction_mode='plain')
            content = loader.load()[0].page_content
            chunks = self.text_splitter.split_text(content)
            chunks_num = len(chunks)
        except Exception as e:
            print(f"Error loading or chunking document {doc_file}: {e}")
            chunks = []
            chunks_num = 0

        #print(f"Chunking document: {state.doc_file} into {chunks_num} chunks.")

        return {
            'chunks': chunks,
            'chunks_num': chunks_num,
            'chunk_idx': 0,
            'context': '',
            'queries': queries,
            'doc_file': doc_file,
            'title': state.title,
            'summary': state.summary
        }
    
    @staticmethod
    def _queries2string(queries: List[str]) -> str:
        """
        Converts a list of queries into a single string.
        
        Args:
            queries (List[str]): List of queries to convert.
        
        Returns:
            str: A single string containing all queries separated by new lines.
        """
        return '\n'.join([f'[{i+1}] {query}' for i, query in enumerate(queries)]) if queries else ''
    
    def _get_template_inputs(self, state: ChunkEvalState):
        """
        Prepares the template inputs for the system and task messages.
        """
        prev_chunk = state.chunks[state.chunk_idx - 1] if state.chunk_idx > 0 else ''
        current_chunk = state.chunks[state.chunk_idx] if state.chunk_idx < state.chunks_num else ''
        next_chunk = state.chunks[state.chunk_idx + 1] if state.chunk_idx + 1 < state.chunks_num else ''

        return {
            'title': state.title,
            'summary': state.summary,
            'queries': self._queries2string(state.queries),
            'context': state.context,
            'prev_chunk': prev_chunk,
            'current_chunk': current_chunk,
            'next_chunk': next_chunk,
            'current_chunk_index': state.chunk_idx + 1,
            'total_chunks_num': state.chunks_num
        }
    
    def chunk_eval_node(self, state: ChunkEvalState):
        """
        Evaluates the current chunk based on the query and updates the context.
        """

        template_inputs = self._get_template_inputs(state)
        system_msg = SystemMessage(content=self.prompts['chunk_eval_system'].format(**template_inputs))
        task_msg = HumanMessage(content=self.prompts['chunk_eval_task'].format(**template_inputs))

        eval_result = self.llm.with_structured_output(ChunkEvalAnswer).invoke([system_msg, task_msg]).answer
        labels = [query_eval['evaluation'] for query_eval in sort_dicts_by_id([query.model_dump() for query in eval_result])]

        chunk_eval = ChunkEvalModel(
            idx=state.chunk_idx,
            text=template_inputs['current_chunk'],
            labels=labels
        )

        return {'chunks_eval': [chunk_eval]}
    
    def context_update_node(self, state: ChunkEvalState):
        """
        Updates the context based on the evaluation of the current chunk.
        """
        
        template_inputs = self._get_template_inputs(state)
        system_msg = SystemMessage(content=self.prompts['doc_context_system'].format(**template_inputs))
        update_msg = HumanMessage(content=self.prompts['doc_context_update'].format(**template_inputs))

        context_update = self.llm.invoke([system_msg, update_msg]).content

        return {
            'context': context_update,
            'chunk_idx': state.chunk_idx + 1
        }
    
    def context_aggregation_node(self, state: ChunkEvalState):
        """
        Aggregates current summaries
        """
        template_inputs = self._get_template_inputs(state)
        agg_msg = HumanMessage(content=self.prompts['doc_context_aggregation'].format(**template_inputs))

        context_update = self.llm.invoke([agg_msg]).content

        return {'context': context_update}
    
    def context_aggregation_routing(self, state: ChunkEvalState):
        """
        Determines whether to aggregate context based on the current chunk index.
        If the current chunk index is a multiple of the context aggregation interval and greater than 0,
        it returns 'context_aggregation', otherwise it returns 'chunk_eval_node'.
        """
        if state.chunk_idx % self.context_agg_interval == 0 and state.chunk_idx > 0:
            return 'context_aggregation'
        return 'chunk_eval'
        
    def routing_edge(self, state: ChunkEvalState):
        """
        Determines the next state based on the current chunk index.
        If there are more chunks to evaluate, it returns 'chunk_eval', otherwise it returns 'END'.
        """
        if state.chunk_idx >= state.chunks_num - 1:
            return 'save_chunks_eval'
        return 'context_update'
    
    def save_chunks_eval_node(self, state: ChunkEvalState):
        print(f"Saving evaluation results on document {state.doc_file}...")
        eval_file = os.path.join(self.saving_path, f"{'.'.join(os.path.basename(state.doc_file).split('.')[:-1])}.json")
        with open(eval_file, 'w') as f:
            json.dump({
                'queries': state.queries,
                'doc_file': state.doc_file,
                'title': state.title,
                'chunks_eval': state.chunks_eval
            }, f, indent=4)

        return {'chunk_idx': state.chunk_idx + 1}

    def build_graph(self):
        """
        Builds the state graph for chunk evaluation.
        """
        workflow = StateGraph(ChunkEvalState)
        workflow.add_node('chunking', self.chunking_node)
        workflow.add_node('chunk_eval', self.chunk_eval_node)
        workflow.add_node('context_update', self.context_update_node)
        workflow.add_node('context_aggregation', self.context_aggregation_node)
        workflow.add_node('save_chunks_eval', self.save_chunks_eval_node)

        workflow.add_edge(START, 'chunking')
        workflow.add_edge('chunking', 'chunk_eval')
        workflow.add_conditional_edges('chunk_eval', self.routing_edge, ['context_update', 'save_chunks_eval'])
        workflow.add_conditional_edges('context_update', self.context_aggregation_routing, ['context_aggregation', 'chunk_eval'])
        workflow.add_edge('context_aggregation', 'chunk_eval')
        workflow.add_edge('save_chunks_eval', END)

        self.graph = workflow.compile(checkpointer=self.memory)

    def run(self, metadata_file: str, thread_id: str = None):
        """
        Runs the graph to evaluate chunks of a document based on a query.

        Args:
            metadata_file (str): The path to the metadata file containing queries and document information.
            thread_id (str): Optional thread ID for memory management.
        """
        config = {'configurable': {'thread_id': thread_id}} if thread_id else None
        return self.graph.invoke({
            metadata_file: metadata_file
        }, config=config)
    
    def run_with_progress(self, metadata_file: str, thread_id: str = None):
        """
        Runs the graph with a progress bar to track chunk evaluations.

        Args:
            metadata_file (str): The path to the metadata file containing queries and document information.
            thread_id (str): Optional thread ID for memory management.
        """
        config = {'configurable': {'thread_id': thread_id}} if thread_id else None
        state = self.graph.invoke({
            'metadata_file': metadata_file
        }, 
        config=config,
        interrupt_before='chunk_eval')

        chunks_num = state['chunks_num']
        positive_evaluations = 0
        negative_evaluations = 0

        with tqdm(total=chunks_num, desc="Evaluating Chunks", postfix={"Positive": positive_evaluations, "Negative": negative_evaluations}) as pbar:
            while state['chunk_idx'] < chunks_num:
                # Invoke the next step in the graph
                state = self.graph.invoke(input=None, config={'configurable': {'thread_id': thread_id}}, interrupt_before='chunk_eval')
                
                # Update evaluations
                positive_evaluations = sum([sum([1 for label in chunk['labels'] if label]) for chunk in state['chunks_eval']])
                negative_evaluations = sum([sum([1 for label in chunk['labels'] if not label]) for chunk in state['chunks_eval']])
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(Positive=positive_evaluations, Negative=negative_evaluations)

    async def run_with_progress_async(self, metadata_file: str, thread_id: str = None):
        """
        Runs the graph asynchronously with a progress bar to track chunk evaluations.

        Args:
            metadata_file (str): The path to the metadata file containing queries and document information.
            thread_id (str): Optional thread ID for memory management.

        Returns:
            dict: The final state after all chunks are evaluated.
        """
        config = {'configurable': {'thread_id': thread_id}} if thread_id else None
        state = await asyncio.to_thread(self.graph.invoke, {
            'metadata_file': metadata_file
        }, config=config, interrupt_before='chunk_eval')

        chunks_num = state['chunks_num']
        positive_evaluations = 0
        negative_evaluations = 0

        # Initialize the progress bar
        pbar = notebook_tqdm(total=chunks_num, desc=metadata_file, postfix={"Positive": positive_evaluations, "Negative": negative_evaluations})
        try:
            while state['chunk_idx'] < chunks_num:
                # Invoke the next step in the graph asynchronously
                state = await asyncio.to_thread(self.graph.invoke, input=None, config={'configurable': {'thread_id': thread_id}}, interrupt_before='chunk_eval')

                # Update evaluations
                positive_evaluations = sum([sum([1 for label in chunk['labels'] if label]) for chunk in state['chunks_eval']])
                negative_evaluations = sum([sum([1 for label in chunk['labels'] if not label]) for chunk in state['chunks_eval']])

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(Positive=positive_evaluations, Negative=negative_evaluations)
        finally:
            pbar.close()

        return state
    

from uuid import uuid4

async def run_all_queries_async(chunk_eval_graph, metadata_file_paths):
    """
    Runs run_with_progress_async concurrently for all queries in the metadata file.

    Args:
        chunk_eval_graph (ChunkEvalGraph): The ChunkEvalGraph instance to run evaluations.
        metadata_file_path (str): Path to the metadata file containing queries and document information.

    Returns:
        dict: A dictionary containing the results for each query.
    """

    async def run_query_async(metadata_file: str):
        thread_id = uuid4().hex
        state = await chunk_eval_graph.run_with_progress_async(metadata_file=metadata_file, thread_id=thread_id)
        return metadata_file, state

    # Run all queries concurrently
    tasks = [run_query_async(metadata_file) for metadata_file in metadata_file_paths]
    results = await asyncio.gather(*tasks)