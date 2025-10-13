from google.cloud import run_v2
from google.api_core import exceptions
from geodoc_config import load_config, get_service_config
from google.api_core.operation import Operation

def run_cloud_run_job(job_name: str, env_vars: dict = None):
    """
    Executes a Google Cloud Run job with specified environment variables.

    Args:
        project_id: Your Google Cloud Project ID.
        region: The Google Cloud region where the job is located (e.g., 'us-central1').
        job_name: The name of the Cloud Run job to execute.
        env_vars: A dictionary of environment variable key-value pairs to pass to the job.
                  Defaults to None if no environment variables are provided.

    Returns:
        True if the job execution request was successful and the operation completed
        without immediate errors; False otherwise. Note that this only indicates
        the *request* was successful and the long-running operation finished, not
        necessarily that the job itself completed successfully (check Cloud Run logs
        for job-specific execution success/failure).
    """
    # Load configuration for GCP services and general settings.
    services_config = load_config('gcp_services')
    general_config = load_config('gcp_general')
    try:
        service_config = get_service_config(job_name, "service")
    except:
        service_config = {}
    region = service_config.get("location", services_config['location']) # e.g., us-central1, europe-west1 | use global default if not specified for specific service
    project_id = general_config['project_id']

    # Initialize the Cloud Run Jobs client.
    # This client handles communication with the Cloud Run Admin API.
    client = run_v2.JobsClient()

    # Construct the full resource name for the Cloud Run job.
    # This format is required by the Google Cloud API.
    job_full_name = f"projects/{project_id}/locations/{region}/jobs/{job_name}"

    # Prepare environment variables for the API request.
    # The API expects a list of EnvVar objects.
    env_vars_list = []
    if env_vars:
        for key, value in env_vars.items():
            # Each environment variable is represented by a run_v2.EnvVar object
            # with 'name' and 'value' fields.
            env_vars_list.append(run_v2.EnvVar(name=key, value=value))

    try:
        print(f"Attempting to execute Cloud Run job: '{job_name}'")
        print(f"  Project: {project_id}")
        print(f"  Region: {region}")
        if env_vars:
            print(f"  With environment variables: {env_vars}")
        else:
            print("  No custom environment variables provided.")

        # Construct the RunJobRequest.
        # To pass environment variables, we need to use 'overrides'
        # which specify changes to the job's configuration for this specific execution.
        # 'container_overrides' allows modifying container settings, including env_vars.
        request = run_v2.RunJobRequest(
            name=job_full_name,
            overrides=run_v2.RunJobRequest.Overrides(
                container_overrides=[
                    run_v2.RunJobRequest.Overrides.ContainerOverride(
                        env=env_vars_list
                    )
                ]
            )
        )

        operation = client.run_job(request=request) # Pass the constructed request object

        # Defensive check: Ensure 'operation' is an actual Operation object.
        if not isinstance(operation, Operation):
            print(f"Error: Expected client.run_job to return a valid Operation object, but got {type(operation)}.")
            print("  This might indicate an issue with the client library or the API response.")
            return False

        # Confirmation of initiation based on successful API call and Operation object return.
        print(f"\nCloud Run job '{job_name}' initiation successful.")
        return True 

    except exceptions.NotFound:
        # Handles cases where the specified job, project, or region doesn't exist.
        print(f"Error: Cloud Run Job '{job_name}' not found in project '{project_id}' and region '{region}'.")
        print("Please ensure the job name, project ID, and region are correct.")
        return False
    except exceptions.GoogleAPIError as e:
        # Catches general Google API errors (e.g., permission denied, invalid arguments).
        print(f"An Google Cloud API error occurred: {e.message}")
        print("Please check your permissions and input parameters.")
        return False
    except AttributeError as e: # Catch specific AttributeError for debugging
        print(f"An AttributeError occurred: {e}")
        print("This often means an object did not have an expected attribute (e.g., 'name').")
        print("Please ensure your `google-cloud-run` library is up to date.")
        return False
    except Exception as e:
        # Catches any other unexpected Python errors.
        print(f"An unexpected error occurred: {e}")
        return False