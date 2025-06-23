import sys
sys.path.append("..")
import os
import math
import uuid
from rag_eval import RandomQueriesPaperSearchGraph
from rag_eval import ChunkEvalGraph
from rag_eval.utils import get_filenames_list, get_filenames_without, load_json, remove_files

import asyncio
from tqdm.notebook import tqdm as notebook_tqdm

class ChunkEvalBaseBuilder:
    def __init__(
            self,
            llm,
            output_path: str,
            builder_config: dict = {
                'max_pages': 15,
                'eval_batch_size': 5
            },
            prompts_config: dict = {
                'path': None,
                'random_queries': 'random_queries',
                'paper_queries': 'paper_queries',
                'chunk_eval_system': 'chunk_eval_system',
                'chunk_eval_task': 'chunk_eval_task',
                'doc_context_system': 'doc_context_system',
                'doc_context_update': 'doc_context_update',
                'doc_context_aggregation': 'doc_context_aggregation'
            },
            doc_search_config: dict = {
                'main_queries_num': 4,
                'paper_queries_num': 5,
                'max_results': 5,
            },
            chunk_eval_config: dict = {
                'chunk_size': 600,
                'chunk_overlap': 0,
                'max_queries': 5,
                'context_agg_interval': 5
            },
            memory=None,
            prompt_manager_spec: dict = {}
        ):

        self.llm = llm
        self.output_path = output_path
        self.builder_config = builder_config
        self.prompts_config = prompts_config
        self.doc_search_config = doc_search_config
        self.chunk_eval_config = chunk_eval_config
        self.memory = memory
        self.prompt_manager_spec = prompt_manager_spec

        self.build()

    def build(self):

        self.docs_metadata_path = os.path.join(self.output_path, 'docs_metadata')
        self.docs_path = os.path.join(self.output_path, 'docs')
        self.eval_path = os.path.join(self.output_path, 'chunks_eval')

        self.doc_loader = RandomQueriesPaperSearchGraph(
            llm=self.llm,
            prompts_config=self.prompts_config,
            docs_path=self.docs_path,
            docs_metadata_path=self.docs_metadata_path,
            memory=self.memory,
            prompt_manager_spec=self.prompt_manager_spec,
            **self.doc_search_config
        )

        self.chunk_eval = ChunkEvalGraph(
            llm=self.llm,
            prompts_config=self.prompts_config,
            docs_metadata_path=self.docs_metadata_path,
            saving_path=self.eval_path,
            memory=self.memory,
            prompt_manager_spec=self.prompt_manager_spec,
            **self.chunk_eval_config
        )
    
    def _evaluated_docs(self):
        """
        Returns a list of evaluated documents.
        """
        return get_filenames_list(self.eval_path)

    def _docs_without_evaluation(self):
        """
        Returns a list of documents that have not been evaluated yet.
        """
        return get_filenames_without(self.docs_metadata_path, self._evaluated_docs())
    
    def _all_docs(self):
        """
        Returns a list of all documents.
        """
        return get_filenames_list(self.docs_metadata_path)
    
    @property
    def docs_to_evaluate(self):
        """
        Returns a number of documents that need to be evaluated.
        """
        return len(self._docs_without_evaluation())
    
    @property
    def docs_evaluated(self):
        """
        Returns a number of documents that have been evaluated.
        """
        return len(self._evaluated_docs())
    
    @property
    def all_docs(self):
        """
        Returns a number of all documents.
        """
        return len(self._all_docs())
    
    @property
    def new_docs_per_turn(self):
        return self.doc_search_config['main_queries_num'] * self.doc_search_config['max_results']
    
    @property
    def evaluations_per_turn(self):
        return self.builder_config['eval_batch_size']
    
    def summary(self):
        """
        Prints a summary of the current state of the document loader and evaluator.
        """
        print(f"All docs: {self.all_docs}")
        if self.all_docs > 0:
            print(f"Docs to evaluate: {self.docs_to_evaluate} - {self.docs_to_evaluate/self.all_docs:.2%} of all docs")
            print(f"Docs evaluated: {self.docs_evaluated} - {self.docs_evaluated/self.all_docs:.2%} of all docs")
        print(f"New docs per turn: {self.new_docs_per_turn}")
        print(f"Evaluations per turn: {self.evaluations_per_turn}")
    
    @staticmethod
    def _is_doc_oversized(path, pages_limit):
        doc_metadata = load_json(path)
        return doc_metadata['pages_count'] > pages_limit
    
    @staticmethod
    def _add_json_extension(paths):
        """
        Adds '.json' extension to a list of paths.
        """
        return [f + '.json' for f in paths]
    
    @staticmethod
    def _add_pdf_extension(paths):
        """
        Adds '.pdf' extension to a list of paths.
        """
        return [f + '.pdf' for f in paths]
    
    def _remove_oversized_docs(self):
        """
        Removes oversized documents based on the max_pages limit specified in the builder_config.
        """
        if self.builder_config.get('max_pages'):
            oversized_docs = [
                f for f in self._docs_without_evaluation()
                if self._is_doc_oversized(
                    os.path.join(self.docs_metadata_path, f + '.json'),
                    self.builder_config['max_pages']
                )
            ]
            print(f"Removing oversized docs: {oversized_docs}")
            docs_before = self.all_docs
            remove_files(self.docs_metadata_path, self._add_json_extension(oversized_docs))
            remove_files(self.docs_path, self._add_pdf_extension(oversized_docs))
            docs_after = self.all_docs
            docs_removed = docs_before - docs_after
            print(f"Removed {docs_removed} oversized docs. Remaining docs: {docs_after}")

    def extend_docs(self, target_docs: int):

        current_docs_num = self.all_docs
        docs_per_turn = self.new_docs_per_turn

        turns_needed = math.ceil((target_docs - current_docs_num) / docs_per_turn)

        print(f"Current docs: {current_docs_num}, Target docs: {target_docs}, Turns needed: {turns_needed}")

        pbar = notebook_tqdm(total=turns_needed, desc="Collecting docs", postfix={'All docs': current_docs_num, 'New docs': 0})

        for _ in range(turns_needed):

            try:
                thread_id = uuid.uuid4().hex
                self.doc_loader.run(thread_id=thread_id)
            except Exception as e:
                print(f"Error during doc loading: {e}")
            
            all_docs = self.all_docs
            new_docs = all_docs - current_docs_num

            pbar.update(1)
            pbar.set_postfix({'All docs': all_docs, 'New docs': new_docs})

    """async def evaluate_docs(self, target_docs: int):

        self._remove_oversized_docs()

        batch_size = self.evaluations_per_turn
        turns_needed = math.ceil((target_docs - self.docs_evaluated) / batch_size)
        docs_paths = self._add_json_extension(self._docs_without_evaluation())

        async def run_query_async(metadata_file: str):
            thread_id = uuid.uuid4().hex
            state = await self.chunk_eval.run_with_progress_async(metadata_file=metadata_file, thread_id=thread_id)
            return metadata_file, state
        
        process_pbar = notebook_tqdm(total=turns_needed, desc="Evaluating docs", postfix={'Target Docs': target_docs, 'Evaluated': self.docs_evaluated})
        
        for _ in range(turns_needed):

            batch_files = docs_paths[:batch_size]
            docs_paths = docs_paths[batch_size:]

            tasks = [run_query_async(metadata_file) for metadata_file in batch_files]
            results = await asyncio.gather(*tasks)

            process_pbar.set_postfix({'Target Docs': target_docs, 'Evaluated': self.docs_evaluated})"""
    
    async def evaluate_docs(self, target_docs: int):
        self._remove_oversized_docs()

        batch_size = self.evaluations_per_turn
        docs_needed = max(target_docs - self.docs_evaluated, 0)
        docs_paths = self._add_json_extension(self._docs_without_evaluation())[:docs_needed]
        total_docs = len(docs_paths)

        # Progress bar
        process_pbar = notebook_tqdm(total=total_docs, desc="Evaluating docs", postfix={'Target Docs': target_docs, 'Evaluated': self.docs_evaluated})

        async def worker(queue):
            while True:
                metadata_file = await queue.get()
                if metadata_file is None:  # Sentinel to stop the worker
                    break
                try:
                    thread_id = uuid.uuid4().hex
                    state = await self.chunk_eval.run_with_progress_async(metadata_file=metadata_file, thread_id=thread_id)
                except Exception as e:
                    print(f"Error processing {metadata_file}: {e}")
                finally:
                    process_pbar.update(1)
                    process_pbar.set_postfix({'Target Docs': target_docs, 'Evaluated': self.docs_evaluated})
                    queue.task_done()

        # Create a queue and populate it with files to process
        queue = asyncio.Queue()
        for metadata_file in docs_paths:
            await queue.put(metadata_file)

        # Start worker tasks
        num_workers = batch_size  # Number of concurrent workers
        workers = [asyncio.create_task(worker(queue)) for _ in range(num_workers)]


        # Wait for all tasks to complete
        await queue.join()

        # Stop workers
        for _ in range(num_workers):
            await queue.put(None)
        await asyncio.gather(*workers)