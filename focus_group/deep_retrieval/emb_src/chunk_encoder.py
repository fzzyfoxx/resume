from rag_eval import ChunkDataHandler

import tensorflow_hub as hub
import tensorflow_text as tftext
import tensorflow as tf
import sys
sys.path.append("..")
import numpy as np

def chunk2query_cosine_similarity(encoded_chunks, encoded_queries):
    K_norm = tf.norm(encoded_chunks, axis=-1, keepdims=False)
    Q_norm = tf.norm(encoded_queries, axis=-1, keepdims=False)
    return tf.matmul(encoded_chunks, encoded_queries, transpose_b=True) / (tf.expand_dims(Q_norm, axis=0) * tf.expand_dims(K_norm, axis=1))

class ChunkEncoder:
    def __init__(
            self,
            data_path: str = '../data',
            preprocessor_path: str = '../models/preprocessor/en_uncased_preprocess',
            encoder_path: str = '../models/encoder/cmlm-en-base',
            seq_length: int = 256
    ):
        self.data_path = data_path
        self.preprocessor_path = preprocessor_path
        self.encoder_path = encoder_path
        self.seq_length = seq_length

        self.build()

    def build(self):
        self.data_handler = ChunkDataHandler(output_path=self.data_path)
        self.preprocessor = hub.load(self.preprocessor_path)
        self.tokenizer = hub.KerasLayer(self.preprocessor.tokenize)
        self.inputer = hub.KerasLayer(self.preprocessor.bert_pack_inputs, 
                                      arguments=dict(seq_length=self.seq_length))
        self.encoder = hub.KerasLayer(self.encoder_path)

    def get_doc_names(self):
        return self.data_handler._evaluated_docs()
    
    def get_doc_titles(self, doc_names):
        return self.data_handler.get_doc_titles(doc_names)
    
    @staticmethod
    def check_max_token_length(encoder_inputs):
        return tf.reduce_max(tf.reduce_sum(encoder_inputs['input_mask'], axis=1)).numpy()
    
    def load_doc(self, doc_name):
        eval_metadata = self.data_handler.get_eval_file(doc_name)
        chunks = [chunk_spec['text'] for chunk_spec in eval_metadata['chunks_eval']]
        chunks_labels = tf.constant([chunk_spec['labels'] for chunk_spec in eval_metadata['chunks_eval']], tf.int32)
        queries = eval_metadata['queries']
        return chunks, chunks_labels, queries
    
    def prepare_inputs(self, chunks, queries):
        tokenized_inputs = self.tokenizer(chunks + queries)
        encoder_inputs = self.inputer([tokenized_inputs])
        queries_num = len(queries)
        return encoder_inputs, queries_num
    
    def encode_inputs(self, inputs, queries_num):
        encoder_output = self.encoder(inputs)['pooled_output']
        encoded_chunks = encoder_output[:-queries_num]
        encoded_queries = encoder_output[-queries_num:]
        return encoded_chunks, encoded_queries
    
    def encode_doc(self, doc_name, return_all=False):
        chunks, chunks_labels, queries = self.load_doc(doc_name)
        inputs, queries_num = self.prepare_inputs(chunks, queries)
        encoded_chunks, encoded_queries = self.encode_inputs(inputs, queries_num)
        if return_all:
            return {
                'encoded_chunks': encoded_chunks,
                'encoded_queries': encoded_queries,
                'chunks_labels': chunks_labels,
                'queries': queries,
                'chunks': chunks,
            }
        return {
            'encoded_chunks': encoded_chunks,
            'encoded_queries': encoded_queries,
            'chunks_labels': chunks_labels,
        }
    
    def evaluate_similarities(self, max_docs=None, threshold=0.45):
        doc_names = self.get_doc_names()
        if max_docs is not None:
            doc_names = doc_names[:max_docs]

        f1_score = tf.keras.metrics.F1Score(threshold=threshold, average='macro')
        precision = tf.keras.metrics.Precision(thresholds=threshold)
        recall = tf.keras.metrics.Recall(thresholds=threshold)
        similarity_coverage = tf.keras.metrics.Mean(name='similarity_coverage')
        labels_coverage = tf.keras.metrics.Mean(name='labels_coverage')

        pb = tf.keras.utils.Progbar(len(doc_names), stateful_metrics=['f1_score', 'precision', 'recall', 'similarity_coverage', 'labels_coverage'])
        
        for i, doc_name in enumerate(doc_names):
            encodings = self.encode_doc(doc_name, return_all=False)
            similarity = chunk2query_cosine_similarity(encodings['encoded_chunks'], encodings['encoded_queries'])
            labels = tf.cast(encodings['chunks_labels'], tf.float32)

            f1_score.update_state(labels, similarity)
            precision.update_state(labels, similarity)
            recall.update_state(labels, similarity)

            similarity_coverage.update_state(tf.reduce_mean(tf.where(similarity> threshold, 1.0, 0.0)))
            labels_coverage.update_state(tf.reduce_mean(labels))

            pb.update(i+1, values=[('f1_score', f1_score.result().numpy()),
                                ('precision', precision.result().numpy()),
                                ('recall', recall.result().numpy()),
                                ('similarity_coverage', similarity_coverage.result().numpy()),
                                ('labels_coverage', labels_coverage.result().numpy())
                                ])

        # Return the evaluation metrics
        print("\nEvaluation Results:")
        print(f"F1 Score: {f1_score.result().numpy():.2f}")
        print(f"Precision: {precision.result().numpy():.2f}")
        print(f"Recall: {recall.result().numpy():.2f}")
        print(f"Similarity Coverage: {similarity_coverage.result().numpy():.2%}")
        print(f"Labels Coverage: {labels_coverage.result().numpy():.2%}")

        return {
            'f1_score': f1_score.result().numpy(),
            'precision': precision.result().numpy(),
            'recall': recall.result().numpy(),
            'similarity_coverage': similarity_coverage.result().numpy(),
            'labels_coverage': labels_coverage.result().numpy()
        }
    

class HyDEChunkEncoder(ChunkEncoder):
    def __init__(
            self,
            data_path: str = '../data',
            preprocessor_path: str = '../models/preprocessor/en_uncased_preprocess',
            encoder_path: str = '../models/encoder/cmlm-en-base',
            seq_length: int = 256
    ):
        super().__init__(
            data_path=data_path,
            preprocessor_path=preprocessor_path,
            encoder_path=encoder_path,
            seq_length=seq_length
        )

    def get_doc_names(self):
        return self.data_handler._docs_with_hyde()
    
    def load_hyde_doc(self, doc_name):
        chunks, chunks_labels, _ = self.load_doc(doc_name)
        hyde_doc = self.data_handler.get_hyde_file(doc_name)

        queries_lengths = [len(hyde_set['hyde_queries'])+1 for hyde_set in hyde_doc['queries_set']]
        concatenated_queries = [query for queries_set in hyde_doc['queries_set'] for query in [queries_set['query']] + queries_set['hyde_queries']]

        return chunks, chunks_labels, concatenated_queries, queries_lengths
    
    def encode_doc(self, doc_name, return_all=False):
        chunks, chunks_labels, queries, queries_lengths = self.load_hyde_doc(doc_name)
        inputs, queries_num = self.prepare_inputs(chunks, queries)
        encoded_chunks, encoded_queries = self.encode_inputs(inputs, queries_num)
        if return_all:
            return {
                'encoded_chunks': encoded_chunks,
                'encoded_queries': encoded_queries,
                'chunks_labels': chunks_labels,
                'queries': queries,
                'queries_lengths': queries_lengths,
                'chunks': chunks,
            }
        return {
            'encoded_chunks': encoded_chunks,
            'encoded_queries': encoded_queries,
            'chunks_labels': chunks_labels,
            'queries_lengths': queries_lengths
        }
    
    def aggregate_hyde_queries(self, results, queries_lengths):
        results_set = tf.split(results, num_or_size_splits=queries_lengths, axis=-1)
        return tf.concat([tf.reduce_max(x, axis=-1, keepdims=True) for x in results_set], axis=-1)
    
    def evaluate_similarities(self, max_docs=None, threshold=0.45):
        doc_names = self.get_doc_names()
        if max_docs is not None:
            doc_names = doc_names[:max_docs]

        f1_score = tf.keras.metrics.F1Score(threshold=threshold, average='macro')
        precision = tf.keras.metrics.Precision(thresholds=threshold)
        recall = tf.keras.metrics.Recall(thresholds=threshold)
        similarity_coverage = tf.keras.metrics.Mean(name='similarity_coverage')
        labels_coverage = tf.keras.metrics.Mean(name='labels_coverage')

        pb = tf.keras.utils.Progbar(len(doc_names), stateful_metrics=['f1_score', 'precision', 'recall', 'similarity_coverage', 'labels_coverage'])
        
        for i, doc_name in enumerate(doc_names):
            encodings = self.encode_doc(doc_name, return_all=False)
            similarity = chunk2query_cosine_similarity(encodings['encoded_chunks'], encodings['encoded_queries'])
            labels = tf.cast(encodings['chunks_labels'], tf.float32)

            similarity = self.aggregate_hyde_queries(similarity, encodings['queries_lengths'])

            f1_score.update_state(labels, similarity)
            precision.update_state(labels, similarity)
            recall.update_state(labels, similarity)

            similarity_coverage.update_state(tf.reduce_mean(tf.where(similarity> threshold, 1.0, 0.0)))
            labels_coverage.update_state(tf.reduce_mean(labels))

            pb.update(i+1, values=[('f1_score', f1_score.result().numpy()),
                                ('precision', precision.result().numpy()),
                                ('recall', recall.result().numpy()),
                                ('similarity_coverage', similarity_coverage.result().numpy()),
                                ('labels_coverage', labels_coverage.result().numpy())
                                ])

        # Return the evaluation metrics
        print("\nEvaluation Results:")
        print(f"F1 Score: {f1_score.result().numpy():.2f}")
        print(f"Precision: {precision.result().numpy():.2f}")
        print(f"Recall: {recall.result().numpy():.2f}")
        print(f"Similarity Coverage: {similarity_coverage.result().numpy():.2%}")
        print(f"Labels Coverage: {labels_coverage.result().numpy():.2%}")

        return {
            'f1_score': f1_score.result().numpy(),
            'precision': precision.result().numpy(),
            'recall': recall.result().numpy(),
            'similarity_coverage': similarity_coverage.result().numpy(),
            'labels_coverage': labels_coverage.result().numpy()
        }