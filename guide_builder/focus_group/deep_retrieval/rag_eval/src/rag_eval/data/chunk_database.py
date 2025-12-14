import sys
sys.path.append("..")
import os
import math
import uuid
from rag_eval import RandomQueriesPaperSearchGraph
from rag_eval import ChunkEvalGraph
from rag_eval.utils import get_filenames_list, get_filenames_without, load_json, remove_files, create_path_if_not_exists

import asyncio
from tqdm.notebook import tqdm as notebook_tqdm

class ChunkDataHandler:
    """
    A class for managing and handling document chunks for evaluation.

    This class provides methods for organizing, retrieving, and summarizing document metadata, 
    evaluation files, and HyDE queries. It also tracks the state of document evaluation and 
    provides utilities for managing document paths.

    Attributes:
        output_path: The base directory for storing document metadata, evaluation files, and HyDE queries.
        docs_metadata_path: The directory for storing document metadata files.
        docs_path: The directory for storing document files.
        eval_path: The directory for storing evaluation files.
        hyde_path: The directory for storing HyDE queries.
    """

    def __init__(
            self,
            output_path: str,
        ):
        """
        Initialize the ChunkDataHandler instance.

        Args:
            output_path: The base directory for storing document-related files.
        """
        self.output_path = output_path
        create_path_if_not_exists(self.output_path)

        self.docs_metadata_path = os.path.join(self.output_path, 'docs_metadata')
        self.docs_path = os.path.join(self.output_path, 'docs')
        self.eval_path = os.path.join(self.output_path, 'chunks_eval')
        self.hyde_path = os.path.join(self.output_path, 'hyde_queries')

    def _evaluated_docs(self):
        """
        Returns a list of evaluated documents.

        Returns:
            List[str]: A list of filenames for evaluated documents.
        """
        return get_filenames_list(self.eval_path)

    def _docs_without_evaluation(self):
        """
        Returns a list of documents that have not been evaluated yet.

        Returns:
            List[str]: A list of filenames for documents without evaluation.
        """
        return get_filenames_without(self.docs_metadata_path, self._evaluated_docs())
    
    def _docs_with_hyde(self):
        """
        Returns a list of documents that have HyDE queries saved.

        Returns:
            List[str]: A list of filenames for documents with HyDE queries.
        """
        return get_filenames_list(self.hyde_path)
    
    def _docs_without_hyde(self):
        """
        Returns a list of documents that do not have HyDE queries saved but have evaluated chunks.

        Returns:
            List[str]: A list of filenames for documents without HyDE queries.
        """
        return get_filenames_without(self.eval_path, self._docs_with_hyde())
    
    def _all_docs(self):
        """
        Returns a list of all documents.

        Returns:
            List[str]: A list of filenames for all documents.
        """
        return get_filenames_list(self.docs_metadata_path)
    
    def get_eval_file(self, doc_name: str):
        """
        Load the evaluation file for a given document name.

        Args:
            doc_name: The name of the document.

        Returns:
            dict: The content of the evaluation file.

        Raises:
            FileNotFoundError: If the evaluation file does not exist.
        """
        path = os.path.join(self.eval_path, f"{doc_name}.json")
        if os.path.exists(path):
            return load_json(path)
        else:
            raise FileNotFoundError(f"Evaluation file for document '{doc_name}' not found at {self.eval_path}.")
        
    def get_metadata_file(self, doc_name: str):
        """
        Load the metadata file for a given document name.

        Args:
            doc_name: The name of the document.

        Returns:
            dict: The content of the metadata file.

        Raises:
            FileNotFoundError: If the metadata file does not exist.
        """
        path = os.path.join(self.docs_metadata_path, f"{doc_name}.json")
        if os.path.exists(path):
            return load_json(path)
        else:
            raise FileNotFoundError(f"Metadata file for document '{doc_name}' not found at {self.docs_metadata_path}.")
        
    def get_hyde_file(self, doc_name: str):
        """
        Load the HyDE queries file for a given document name.

        Args:
            doc_name: The name of the document.

        Returns:
            dict: The content of the HyDE queries file.

        Raises:
            FileNotFoundError: If the HyDE queries file does not exist.
        """
        path = os.path.join(self.hyde_path, f"{doc_name}.json")
        if os.path.exists(path):
            return load_json(path)
        else:
            raise FileNotFoundError(f"HyDE queries file for document '{doc_name}' not found at {self.hyde_path}.")
        
    def get_doc_titles(self, docs_names: list):
        """
        Retrieve the titles of the given documents.

        Args:
            docs_names: A list of document names.

        Returns:
            List[str]: A list of document titles.
        """
        titles = []
        for doc_name in docs_names:
            metadata = self.get_metadata_file(doc_name)
            titles.append(metadata.get('title', 'Unknown Title'))
        return titles
    
    def summary(self):
        """
        Print a summary of the current state of the document loader and evaluator.
        """
        print(f"All docs: {self.all_docs}")
        if self.all_docs > 0:
            print(f"Docs to evaluate: {self.docs_to_evaluate} - {self.docs_to_evaluate/self.all_docs:.2%} of all docs")
            evaluated_docs = self.docs_evaluated
            print(f"Docs evaluated: {evaluated_docs} - {evaluated_docs/self.all_docs:.2%} of all docs")
            if evaluated_docs > 0 & os.path.exists(self.hyde_path):
                print(f"Docs with HyDE queries: {len(self._docs_with_hyde())} - {len(self._docs_with_hyde())/evaluated_docs:.2%} of evaluated docs")
    
    @property
    def docs_to_evaluate(self):
        """
        Get the number of documents that need to be evaluated.

        Returns:
            int: The number of documents to evaluate.
        """
        return len(self._docs_without_evaluation())
    
    @property
    def docs_evaluated(self):
        """
        Get the number of documents that have been evaluated.

        Returns:
            int: The number of evaluated documents.
        """
        return len(self._evaluated_docs())
    
    @property
    def all_docs(self):
        """
        Get the total number of documents.

        Returns:
            int: The total number of documents.
        """
        return len(self._all_docs())
    
    @staticmethod
    def _add_json_extension(paths):
        """
        Add the '.json' extension to a list of paths.

        Args:
            paths: A list of file paths.

        Returns:
            List[str]: The file paths with the '.json' extension added.
        """
        return [f + '.json' for f in paths]
    
    @staticmethod
    def _add_pdf_extension(paths):
        """
        Add the '.pdf' extension to a list of paths.

        Args:
            paths: A list of file paths.

        Returns:
            List[str]: The file paths with the '.pdf' extension added.
        """
        return [f + '.pdf' for f in paths]


class ChunkEvalBaseBuilder(ChunkDataHandler):
    """
    A class for building and managing chunk evaluation workflows.

    This class extends ChunkDataHandler and provides additional functionality for 
    loading documents, evaluating chunks, and managing configurations for the evaluation process.

    Attributes:
        llm: The language model used for generating responses.
        builder_config: Configuration for the builder, including batch sizes and limits.
        prompts_config: Configuration for prompts used in the evaluation process.
        doc_search_config: Configuration for document search parameters.
        chunk_eval_config: Configuration for chunk evaluation parameters.
        memory: Optional memory object for state persistence.
        prompt_manager_spec: Specifications for managing prompts.
    """

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
        """
        Initialize the ChunkEvalBaseBuilder instance.

        Args:
            llm: The language model used for generating responses.
            output_path: The base directory for storing document-related files.
            builder_config: Configuration for the builder, including batch sizes and limits.
            prompts_config: Configuration for prompts used in the evaluation process.
            doc_search_config: Configuration for document search parameters.
            chunk_eval_config: Configuration for chunk evaluation parameters.
            memory: Optional memory object for state persistence.
            prompt_manager_spec: Specifications for managing prompts.
        """
        super().__init__(
            output_path=output_path
            )

        self.llm = llm
        self.builder_config = builder_config
        self.prompts_config = prompts_config
        self.doc_search_config = doc_search_config
        self.chunk_eval_config = chunk_eval_config
        self.memory = memory
        self.prompt_manager_spec = prompt_manager_spec

        self.build()

    def build(self):
        """
        Build the document loader and chunk evaluator based on the provided configurations.
        """
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
    
    @property
    def new_docs_per_turn(self):
        """
        Get the number of new documents added per turn.

        Returns:
            int: The number of new documents per turn.
        """
        return self.doc_search_config['main_queries_num'] * self.doc_search_config['max_results']
    
    @property
    def evaluations_per_turn(self):
        """
        Get the number of evaluations performed per turn.

        Returns:
            int: The number of evaluations per turn.
        """
        return self.builder_config['eval_batch_size']
    
    def summary(self):
        """
        Print a summary of the current state of the document loader and evaluator.
        """
        print(f"All docs: {self.all_docs}")
        if self.all_docs > 0:
            print(f"Docs to evaluate: {self.docs_to_evaluate} - {self.docs_to_evaluate/self.all_docs:.2%} of all docs")
            print(f"Docs evaluated: {self.docs_evaluated} - {self.docs_evaluated/self.all_docs:.2%} of all docs")
        print(f"New docs per turn: {self.new_docs_per_turn}")
        print(f"Evaluations per turn: {self.evaluations_per_turn}")
    
    @staticmethod
    def _is_doc_oversized(path, pages_limit):
        """
        Check if a document exceeds the maximum allowed number of pages.

        Args:
            path: The path to the document metadata file.
            pages_limit: The maximum number of pages allowed.

        Returns:
            bool: True if the document is oversized, False otherwise.
        """
        doc_metadata = load_json(path)
        return doc_metadata['pages_count'] > pages_limit
    
    def _remove_oversized_docs(self):
        """
        Remove oversized documents based on the max_pages limit specified in the builder_config.
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
        """
        Extend the document collection to reach the target number of documents.

        Args:
            target_docs: The target number of documents.
        """
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

    async def evaluate_docs(self, target_docs: int):
        """
        Evaluate the documents to reach the target number of evaluated documents.

        Args:
            target_docs: The target number of evaluated documents.
        """
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