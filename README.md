
# Frameworks and Libraries Usage Examples
#### TensorFlow
- [Chunk embeddings](guide_builder/focus_group/deep_retrieval/notebooks/embeddings.ipynb) - Usage of `TensorFlow Hub` and `TensorFlow Text` to generate text embeddings for document chunks.

#### KerasTuner

#### MLFlow

#### scikit-learn
- [PCA](guide_builder/focus_group/deep_retrieval/notebooks/embeddings.ipynb) - Principal Component Analysis for representation of document's chunks embeddings.
- [Spectral Clustering](guide_builder/focus_group/deep_retrieval/notebooks/chunks_relations.ipynb) - Chunks clustering based on embeddings cosine similarity matrix.
- [TF-IDF](guide_builder/focus_group/deep_retrieval/notebooks/lang_chunk_clustering.ipynb) - Text vectorization using Term Frequency-Inverse Document Frequency for chunk retriever.

#### SciPy
- [Eigenvalue Spectrum of Laplacian](guide_builder/focus_group/deep_retrieval/notebooks/chunks_relations.ipynb) - Finding optimal number of clusters using eigenvalue spectrum plot.
- [Voronoi diagram](guide_builder/pure_llm_gen/src/cover.py) - Generating Voronoi diagram for random points to create abstract book cover images.

#### NumPy
- [Voronoi cover](guide_builder/pure_llm_gen/src/cover.py) - Handle voronoi polygons to generate matching color fills for book cover images.

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

#### LangGraph

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