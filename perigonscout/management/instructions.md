# OPERATIONS

## SETUP

Prepare the environment for the GeoDoc services. This includes setting up Google Cloud Storage buckets and BigQuery tables.
```bash
geodoc-setup-gcp-for-service --service SERVICE_NAME
```
- SERVICE_NAME: The name of the service to set up (e.g., bdot).

## DEPLOYMENT

### SAVE PACKAGES

Before building docker images, ensure that all required packages are saved to wheel files within geodoc_deploy/packages.
This can be done by running the following command:
```bash
sh management/packages/save_packages.sh
```

### BUILD IMAGE

To build a Docker image for a specific service, use the following command:
```bash
geodoc-build-image --service SERVICE_NAME
```
- SERVICE_NAME: The name of the service to build (e.g., bdot).

### PUSH IMAGE
To push the built Docker image to Google Container Registry, use:
```bash
geodoc-deploy-image-gcp --service SERVICE_NAME
```
- SERVICE_NAME: The name of the service to deploy (e.g., bdot).

### CREATE CloudRun JOB
To create a CloudRun job for a specific service, use:
```bash
geodoc-create-job --service SERVICE_NAME
```
- SERVICE_NAME: The name of the service to create a job for (e.g., bdot).

### FULL DEPLOYMENT
To deploy a service fully, including building the image and creating the CloudRun job, use:
```bash
geodoc-add-service --service SERVICE_NAME
```
- SERVICE_NAME: The name of the service to deploy (e.g., bdot).

### PREPARE GRID
To prepare the grid table with specific size use:
```bash
geodoc-prepare-grid --grid_size GRID_SIZE --min_side_buffer MIN_SIDE_BUFFER --delete_local DELETE_LOCAL
```
- GRID_SIZE: The size of the grid cells (e.g., 1000).
- MIN_SIDE_BUFFER: The minimum buffer for the source shape.
- DELETE_LOCAL(INT): Whether to delete local files after processing (default: 1).

## RUNNING

### ADD TO QUEUE

To add items to a queue for processing within specific services, use the following command:
```bash
geodoc-add-to-queue --service SERVICE_NAME --source_type SOURCE --source_key SOURCE_KEY --teryt_pattern TERYT_PATTERN --priority PRIORITY
```

- SERVICE_NAME: The name of the service to which the item should be added (e.g., bdot).
- SOURCE: The type of source from which the item (like teryt) is being added (default: administration_unit). The name has to be included in configs_map.json.
- SOURCE_KEY: The table of the source item (e.g., county).
- TERYT_PATTERN: The pattern for the TERYT code (e.g., '0201').
- PRIORITY: The priority of the item in the queue (default: 1). Higher numbers indicate higher priority.

### RUN LOCAL SERVICE

To run a service locally, use the following command:
```bash
geodoc-run-local-job --job SERVICE_NAME --e ENV_VAR1 --e ENV_VAR2
``` 

- SERVICE_NAME: The name of the service to run (e.g., bdot).
- ENV_VAR1, ENV_VAR2: Environment variables to set for the service (optional).

### RUN GCP SERVICE

To run a service on Google Cloud Platform (GCP), use the following command:
```bash
geodoc-run-job --job SERVICE_NAME --e ENV_VAR1 --e ENV_VAR2
``` 

- SERVICE_NAME: The name of the service to run on GCP (e.g., bdot).
- ENV_VAR1, ENV_VAR2: Environment variables to set for the service (optional).

# SERVICES

## SPATIAL DATA DOWNLOADER

Multi purpose service for handling different data sources downloaded based on TERYT codes.
The spatial-data-downloader is used only in deployment and run phase. For a setup and add-to-queue operations use supported services names.

```bash
geodoc-run-job --job spatial-data-downloader --e SERVICE=SERVICE_NAME --e QUEUE_LIMIT=QUEUE_LIMIT
```
- SERVICE_NAME: The name of the service to run (e.g., bdot).
- QUEUE_LIMIT: The maximum number of items to process in the queue (default: 5).

### SUPPORTED SERVICES
- bdot: Downloads and processes BDOT data for counties.
- soil-complexes: Downloads and processes soil complexes data based on a grid.
- parcels: Downloads and processes parcel data based on a grid.

## PARCELS

Separate instance of the spatial-data-downloader service to handle multi-worker processing of parcels data.
Designed for CloudRun jobs, running it locally will not launch multiple workers.
```bash
geodoc-run-job --job parcels --e QUEUE_LIMIT=QUEUE_LIMIT --e QUEUE_SPLIT_METHOD=QUEUE_SPLIT_METHOD
```
- QUEUE_LIMIT: The maximum number of items to process in the queue (default: 5).
- QUEUE_SPLIT_METHOD: The method to split the queue items (default: cut). Options are:
  - cut: Splits the queue items into equal parts for each worker by separating a list like [1,2], [3,4], [5,6].
  - modulo: Splits the queue items based on the modulo operation, allowing for more flexible distribution across workers like [1,4], [2,5], [3,6].

## EJOURNALS-DOWNLOADER
Service for downloading documents from province's e-journals.
```bash
geodoc-run-job --job ejournals-downloader --e QUEUE_LIMIT=QUEUE_LIMIT --e FILTER=FILTER --e TEXT_FILTER=TEXT_FILTER --e QUEUE_SPLIT_METHOD=QUEUE_SPLIT_METHOD
```

- QUEUE_LIMIT: The maximum number of items to process in the queue (default: 8).
- FILTER: A query "where" clause to filter queue (default: None) e.g. FILTER="province_id='30' AND year=2025".
- TEXT_FILTER: A text string to filter the documents by their title on the e-journals page (default: "zagosp") e.g. TEXT_FILTER="zagosp".
- QUEUE_SPLIT_METHOD: The method to split the queue items (default: cut).