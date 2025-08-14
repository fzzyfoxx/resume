import argparse
from geodoc_deploy.deploy.create_job_src import create_or_update_cloud_run_job
from geodoc_config import get_service_config

def main():
    # Argument parser for command line usage
    parser = argparse.ArgumentParser(description="Deploy a Docker image to Google Artifact Registry.")
    parser.add_argument("--service", type=str, required=True, help="The name of the service to deploy (e.g., 'soil_complexes').")
    args = parser.parse_args()

    # --- Configuration ---
    general_gcp_config = get_service_config("gcp", "general")
    services_config = get_service_config("gcp", "services")

    SERVICE_NAME = args.service
    GCP_PROJECT_ID = general_gcp_config['project_id']
    SERVICE_CONFIG = get_service_config(SERVICE_NAME, "service")['gcp_job']
    ARTIFACT_REGISTRY_LOCATION = services_config['location'] # e.g., us-central1, europe-west1
    ARTIFACT_REGISTRY_REPO = services_config['registry_repository'] # Artifact Registry repository name
    REGION = services_config['location']  # e.g., us-central1, europe-west1

    # Define the full image tag for Artifact Registry
    # Format: <LOCATION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/<IMAGE_NAME>:<TAG>
    IMAGE_NAME = f"{ARTIFACT_REGISTRY_LOCATION}-docker.pkg.dev/{GCP_PROJECT_ID}/{ARTIFACT_REGISTRY_REPO}/{SERVICE_NAME}:latest"

    # Call the function to create/update the job
    create_or_update_cloud_run_job(
        project_id=GCP_PROJECT_ID,
        region=REGION,
        job_name=SERVICE_NAME,
        image_url=IMAGE_NAME,
        config=SERVICE_CONFIG
    )

if __name__ == "__main__":
    main()