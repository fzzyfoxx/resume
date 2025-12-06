## Management

### Overview
This directory contains management `bash` and `python` scripts for setting up and maintaining the Perigon Scout application.

**technological stack** <br>
Bash | BigQuery


### Scripts
- `app/run_local_flask.sh`: A bash script to run the Perigon Scout backend application locally for development and testing purposes.
- `packages/*`: Bash scripts for installation and export of wheel files for Docker image builds.
- `setup/add_columns_values_to_config.py`: A Python script to generate filter values for each CATEGORICAL and SEARCH column in the collections configuration and save them as JSON files in the specified directory structure.
- `setup/setup_container_repo.sh`: A bash script to set up the artifact repository for container images on GCP.