# Projects

### [Maps Vectorization](maps_vectorization)
Map Vectorization is an end-to-end research project that converts dense symbolic maps into structured vector representations using advanced `TensorFlow` workflows. It couples controllable synthetic data generation with scalable `tf.data` pipelines to enable reproducible experiments from compact canvases to full, richly labeled maps. The work introduces **relative radial and rotated positional encodings** that inject query-centric geometry into attention, complemented by frequency-domain modules that remain robust under heavy noise. Architectures span `DETR`, `Mask R-CNN`, `U-Net`, `SegNet`, and lightweight residual backbones, all built as modular `Keras` layers with custom attention, heads, and training utilities. The pipeline supports multi-task supervision (classes, masks, angles, vectors, thickness) alongside specialized losses and metrics, including composite `Hungarian-matching` objectives for detection and instance masks. Configuration-driven tooling assembles datasets, models, and compilers, then trains with `MLflow` to track parameters, artifacts, metrics, and full reproducible configs. Experiments can stream on-the-fly data or consume TFRecords, and seamlessly resume using cached weights and model definitions. Optional hyperparameter search integrates with KerasTuner, while evaluation and plotting reuse the same configs for consistent validation. Together, these components form a practical, research-grade framework for rapid iteration on geometry-aware map vectorization.

<img src="maps_vectorization/resources/pixel_features_rot.png" alt="pixel features" width="900"/>

---

### [PerigonScout](perigonscout)
Perigon Scout is an end-to-end geospatial acquisition and land‑search platform that automates sourcing, normalizing, and serving parcels and related layers from heterogeneous public registries. It combines high‑throughput asynchronous downloaders (`aiohttp`, `asyncio`) with geometry‑aware tiling and grid strategies (`shapely`, `GeoPandas`, `pyproj`, `rasterio`) to assemble county‑ and grid‑based datasets at scale. Containerized services are built and deployed via configuration‑driven CLIs to Google Cloud Run, with images in Artifact Registry, artifacts in GCS, queues in BigQuery/Firestore, and reproducible job specs for local or cloud execution. The loader toolkit includes resilient scrapers for provincial e‑journals (`Selenium`, `BeautifulSoup`), shapefile ingestion, reprojection and concatenation, TIFF/GeoDataFrame handlers, and multi‑worker queue splitting (cut/modulo) for deterministic parallelism. A `Flask` backend (with `Redis` sessions, CORS, and Blueprints) exposes query APIs, while a `React` frontend powers the interactive UI. Utilities cover grid preparation, one‑time bulk uploads, and end‑to‑end deployment (build → push → create job → run) using a consistent CLI. Together, these components deliver a production‑ready pipeline from raw sources to a fast, filterable land‑search experience.

<a href="https://www.youtube.com/watch?v=1o5VWFYmls8">
    <img src="https://img.youtube.com/vi/1o5VWFYmls8/maxresdefault.jpg" alt="Click to watch video" width="50%">
</a>

---

### [Guide Builder](guide_builder)
Guide Builder is a two-tier system for AI-assisted book and guide creation, combining an experiment-friendly agent framework (fcgb) with a streamlined production path (SGB). The fcgb package provides modular `LangGraph` agents, prompt-managed state graphs, and RAG utilities (MongoDB vector search, Tavily web search, fake models) for persona-driven planning, web research, and labeling. The Simplified Guide Builder delivers a complete pipeline from concepts → parts → chapters → contents → intro with configurable prompts, batching, and resumable JSON artifacts, then compiles to PDF via `Markdown2PDF`, including Voronoi-based covers and interactive review widgets. The stack spans `LangChain`/`LangGraph`, `Pydantic`, `PyMuPDF`, `NetworkX`, `scikit-learn`, and `SciPy`, with notebooks for chunk evaluation, HyDE query expansion, spectral clustering, and similarity analysis. Configuration-first design and consistent I/O enable reproducible runs and rapid iteration from toy trials to full books.

<img src="guide_builder/resources/chunks_graph.png" alt="chunks graph" width="450"/>

<br><br>

# Frameworks and Libraries Usage Examples
Here are some examples of how various frameworks and libraries are utilized across different projects in this repository:

#### TensorFlow
- [Chunk embeddings](guide_builder/focus_group/deep_retrieval/notebooks/embeddings.ipynb) - Usage of `TensorFlow Hub` and `TensorFlow Text` to generate text embeddings for document chunks.
- [Custom Metrics](maps_vectorization/models_src/Metrics.py) - Implementation of custom metrics and losses for model training and validation.
- [DETR](maps_vectorization/models_src/DETR.py) - Implementation of Detection Transformer (DETR) architecture for object detection tasks.

much more examples in [Maps Vectorization](maps_vectorization) project documentation.

#### KerasTuner
- [Hyperparameter Tuning](maps_vectorization/models_src/Trainer_support.py) - Subclass of `keras_tuner.HyperModel` for easier experiment management and with MLFlow integration.

#### MLFlow
- [Experiment Tracking Library](maps_vectorization/exp_lib) - Custom library for managing MLFlow experiments, including logging parameters, metrics, and artifacts.

#### scikit-learn
- [PCA](guide_builder/focus_group/deep_retrieval/notebooks/embeddings.ipynb) - Principal Component Analysis for representation of document's chunks embeddings.
- [Spectral Clustering](guide_builder/focus_group/deep_retrieval/notebooks/chunks_relations.ipynb) - Chunks clustering based on embeddings cosine similarity matrix.
- [TF-IDF](guide_builder/focus_group/deep_retrieval/notebooks/lang_chunk_clustering.ipynb) - Text vectorization using Term Frequency-Inverse Document Frequency for chunk retriever.
- [Bayesian Optimization](maps_vectorization/RPN_optimization.ipynb) - Usage of scikit-optimize's `gp_minimize` function for RPN's anchors size optimization.
- [K-Means with TensorFlow](maps_vectorization/models_src/Kmeans.py) - Implementation of K-Means clustering algorithm integrated into TensorFlow pipeline.

#### SciPy
- [Eigenvalue Spectrum of Laplacian](guide_builder/focus_group/deep_retrieval/notebooks/chunks_relations.ipynb) - Finding optimal number of clusters using eigenvalue spectrum plot.
- [Voronoi diagram](guide_builder/pure_llm_gen/src/cover.py) - Generating Voronoi diagram for random points to create abstract book cover images.

#### NumPy
- [Voronoi cover](guide_builder/pure_llm_gen/src/cover.py) - Handle voronoi polygons to generate matching color fills for book cover images.
- [Vector Dataset](maps_vectorization/models_src/VecDataset.py) - Randomized pattern generation for training dataset preparation.

#### Pandas
- [GeoDataFrame preprocessing](perigonscout/geodoc_loader/geodoc_loader/handlers/tiff.py) - Transformation of TIFF files into GeoDataFrames.
- [Table styling](guide_builder/pure_llm_gen/src/display.py) - Creating styled tables for `ipywidgets' display.

#### asyncio
- [Asynchronous requests](perigonscout/geodoc_loader/geodoc_loader/services/uldk.py) - Making asynchronous HTTP requests to ULDK API for fetching geospatial data with usage of `aiohttp` and `asyncio`.
- [Parallel chunk evaluation](guide_builder/focus_group/deep_retrieval/rag_eval/src/rag_eval/data.py) - Evaluating multiple document chunks with LLM in parallel using `asyncio.Queue` function for improved performance.

#### Flask
- [Backend Service Setup](perigonscout/app_source/backend/app/__init__.py) - Setting up a Flask application with `Redis` for session management, `CORS` for handling cross-origin requests and `Blueprints` for modularizing the application structure.
- [Requests Handling](perigonscout/app_source/backend/app/routes/queries.py) - Defining routes to handle incoming `GET` and `POST` requests for querying geospatial data and returning results in JSON format.

#### Shapely
- [Grid for polygon](perigonscout/geodoc_loader/geodoc_loader/services/grid.py) - Creating a rectangular grid over a polygon with respect to given `CRS`.
- [Vector Data API Acquisition](perigonscout/geodoc_loader/geodoc_loader/services/uldk.py) - Highly optimized algorithm for fetching vector data to fill complex geometries with acquired polygons based on point inquiries.
- [Projection Transformations](perigonscout/geodoc_loader/geodoc_loader/handlers/geom.py) - Performing projection transformations on geometries with cooperation of `PyProj` library.

#### GeoPandas
- [Shapefiles handling](perigonscout/geodoc_loader/geodoc_loader/handlers/core.py) - Reading and preprocessing of shapefiles using `GeoDataFrame` feature.

#### OpenCV
- [Symbolic maps generation](maps_vectorization/src) - Complex patterns drawing to create symbolic map images representing various geographical features.

#### LangGraph
- [Research Agent](guide_builder/focus_group/fcgb/src/fcgb/rag/researcher.py) - Implementation of a research agent that uses `vector stores`, `retrievers` and conducts self-guided web searches to gather information and answer complex queries.
- [ChatBot Classes](guide_builder/focus_group/fcgb/src/fcgb/chatbots/chatbot.py) - Base classes for managing chatbot interactions, including message handling, conversation history, and response generation using LLMs.
- [Job Handler](guide_builder/focus_group/fcgb/src/fcgb/tools/job_handler.py) - An agent designed to solve tasks with usage of given tools.

#### LangChain
- [Chains usage](guide_builder/pure_llm_gen/src/builder.py) - Implementation of custom chains for generating book content using `RunnableParallel` and `RunnableLambda` functions.

#### MongoDB
- [RAG Handler Class](guide_builder/focus_group/fcgb/src/fcgb/rag/mongodb.py) - A class for managing MongoDB-based Retrieval-Augmented Generation (RAG) workflows.This class provides methods for creating vector search indexes, adding documents with embeddings, and retrieving documents based on vector similarity.
- [Local server](guide_builder/focus_group/databases/mongodb/docker-compose.yml) - Docker Compose configuration for setting up a local MongoDB server with authentication.
- [Vector Search](guide_builder/focus_group/experimental_notebooks/researcher.ipynb) - Example of using MongoDB's vector search capabilities to retrieve relevant documents based on text embeddings.

#### Google Cloud Platform (GCP)
- [Cloud Run Job Management](perigonscout/geodoc_deploy/geodoc_deploy/deploy/create_job_src.py) - Creating or updating Cloud Run jobs with custom configurations such as task count, CPU, memory, environment variables, and timeout settings.
- [Artifact Registry Authentication](perigonscout/geodoc_deploy/geodoc_deploy/deploy/deploy_image_gcp_src.py) - Authenticating to Google Artifact Registry using service account impersonation and pushing Docker images.
- [E-Journals Setup](perigonscout/geodoc_deploy/geodoc_deploy/setup/ejournals_setup.py) - Preparing GCP resources such as BigQuery datasets, tables, and GCS buckets for e-journal processing workflows.
- [Cloud Run Job Execution](perigonscout/geodoc_run/geodoc_run/src/job.py) - Executing Cloud Run jobs with optional environment variable overrides.
- [BigQuery Queue Management](perigonscout/geodoc_run/geodoc_run/src/queue.py) - Managing processing queues by inserting tasks into BigQuery-backed tables with priority and deduplication.
- [GCS and BigQuery Utilities](perigonscout/geodoc_loader/geodoc_loader/download/gcp.py) - Creating GCS buckets, BigQuery datasets, and tables; uploading GeoJSON files to GCS; and loading data into BigQuery tables.
- [File Downloads from GCS](perigonscout/geodoc_loader/geodoc_loader/data/storage.py) - Downloading files from GCS buckets with MIME type filtering and optional shuffling.