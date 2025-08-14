from geodoc_deploy.deploy.deploy_image_gcp_src import authenticate_and_push_image
from geodoc_config import get_service_config
import argparse

def main():
    # Argument parser for command line usage
    parser = argparse.ArgumentParser(description="Deploy a Docker image to Google Artifact Registry.")
    parser.add_argument("--service", type=str, required=True, help="The name of the service to deploy (e.g., 'soil_complexes').")
    args = parser.parse_args()

    # --- Configuration ---
    # Replace with your actual project ID and service account email
    general_gcp_config = get_service_config("gcp", "general")
    services_config = get_service_config("gcp", "services")

    GCP_PROJECT_ID = general_gcp_config['project_id']
    # This service account needs Artifact Registry Writer permission
    # and the caller of this script needs Service Account Token Creator on it.
    GCP_SERVICE_ACCOUNT_EMAIL = f"{services_config['service_account']}@{GCP_PROJECT_ID}.iam.gserviceaccount.com"
    ARTIFACT_REGISTRY_LOCATION = services_config['location'] # e.g., us-central1, europe-west1
    ARTIFACT_REGISTRY_REPO = services_config['registry_repository'] # Artifact Registry repository name
    SERVICE_NAME = args.service

    # Define the full image tag for Artifact Registry
    # Format: <LOCATION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/<IMAGE_NAME>:<TAG>
    IMAGE_NAME = f"{ARTIFACT_REGISTRY_LOCATION}-docker.pkg.dev/{GCP_PROJECT_ID}/{ARTIFACT_REGISTRY_REPO}/{SERVICE_NAME}:latest"

    # Authenticate and push the image to Artifact Registry
    authenticate_and_push_image(
            image_tag=IMAGE_NAME,
            service_name=SERVICE_NAME,
            service_account_email=GCP_SERVICE_ACCOUNT_EMAIL,
            artifact_registry_host=f"https://{ARTIFACT_REGISTRY_LOCATION}-docker.pkg.dev"
        )

if __name__ == "__main__":
    main()