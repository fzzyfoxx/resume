## geodoc_run package

### Overview
The `geodoc_run` package provides tools for executing data acquisition services within the PerigonScout system.

**technological stack** <br>
Google Cloud Platform | Docker | subprocess

### Endpoints
The package includes command-line interface (CLI) scripts to run specific services:
- geodoc-add-to-queue - adds specified TERYTs codes using provided pattern and a source table to the service queue for processing
- geodoc-run-job - runs a Cloud Run job for a specified service with service-specific environment variables
- geodoc-run-local-job - runs a service job locally