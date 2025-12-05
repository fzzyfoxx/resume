import json
import os
from google.cloud import run_v2
from google.api_core import exceptions
from google.protobuf.duration_pb2 import Duration # Import Duration for timeout

# Ensure you have the Google Cloud SDK installed and authenticated.
# You can authenticate by running:
# gcloud auth application-default login
# Or by setting GOOGLE_APPLICATION_CREDENTIALS environment variable.

def create_or_update_cloud_run_job(
    project_id: str,
    region: str,
    job_name: str,
    image_url: str,
    config: dict # config is now passed directly as a dictionary
):
    """
    Creates or updates a Google Cloud Run job based on a dictionary configuration.

    Args:
        project_id: Your Google Cloud project ID.
        region: The GCP region where the Cloud Run job will be deployed (e.g., 'us-central1').
        job_name: The desired name for the Cloud Run job.
        image_url: The full path to your container image in Artifact Registry
                   (e.g., 'us-central1-docker.pkg.dev/your-project/your-repo/your-image:tag').
        config: A dictionary containing the configuration parameters for the job.
                Expected keys: "taskCount", "cpu", "memory", "env", "maxRetries", "timeoutSeconds".
    """
    client = run_v2.JobsClient()

    # Construct the parent path for the job
    parent = f"projects/{project_id}/locations/{region}"
    job_path = f"{parent}/jobs/{job_name}"

    print(f"Attempting to configure Cloud Run job: {job_name} in {region}...")

    # Determine if the job exists and get its current state if it does
    existing_job = None
    try:
        existing_job = client.get_job(name=job_path)
        job_exists = True
        print(f"Job '{job_name}' already exists. Updating...")
        # Use the existing job object to apply updates
        job_to_configure = existing_job
    except exceptions.NotFound:
        job_exists = False
        print(f"Job '{job_name}' does not exist. Creating...")
        # Create a new job object if it doesn't exist
        job_to_configure = run_v2.Job()
    except Exception as e:
        print(f"An error occurred while checking job existence: {e}")
        return

    # Access the existing (or newly created) JobTemplate and ExecutionTemplate objects
    job_template_obj = job_to_configure.template
    execution_template_obj = job_template_obj.template

    # Ensure the image URL is set. For new jobs, this is straightforward.
    # For existing jobs, we might be updating the image.
    # Always ensure there's at least one container.
    if not execution_template_obj.containers:
        execution_template_obj.containers.append(run_v2.types.Container())
    container = execution_template_obj.containers[0]
    container.image = image_url

    # Apply configuration parameters from JSON
    if "taskCount" in config:
        job_template_obj.task_count = config["taskCount"]
        print(f"  Setting taskCount: {job_template_obj.task_count}")

    # Handle CPU and Memory resources by correctly placing them under 'limits'
    # Ensure container.resources object exists before setting limits
    if not container.resources:
        container.resources = run_v2.types.ResourceRequirements()
    
    # Add to the existing container.resources.limits map
    if "cpu" in config:
        container.resources.limits["cpu"] = config["cpu"]
        print(f"  Setting CPU: {config['cpu']}")
    elif "cpu" in container.resources.limits: # If CPU is removed from config, ensure it's removed from limits
        del container.resources.limits["cpu"]
        print(f"  Removing CPU limit.")

    if "memory" in config:
        container.resources.limits["memory"] = config["memory"]
        print(f"  Setting Memory: {config['memory']}")
    elif "memory" in container.resources.limits: # If Memory is removed from config, ensure it's removed from limits
        del container.resources.limits["memory"]
        print(f"  Removing Memory limit.")
    
    # Handle environment variables
    if "env" in config and isinstance(config["env"], list):
        # Clear existing environment variables to ensure a clean update
        container.env[:] = [] 
        for env_var_data in config["env"]:
            if "name" in env_var_data and ("value" in env_var_data or "valueSource" in env_var_data):
                env_var = run_v2.types.EnvVar()
                env_var.name = env_var_data["name"]
                if "value" in env_var_data:
                    env_var.value = env_var_data["value"]
                elif "valueSource" in env_var_data:
                    # Example for valueSource (e.g., from Secret Manager)
                    # This would be more complex and depend on the exact source structure
                    # env_var.value_source = run_v2.types.EnvVarSource(secret_key_ref=...)
                    print(f"  Warning: valueSource for {env_var_data['name']} is specified but not fully implemented in this example.")
                container.env.append(env_var)
                print(f"  Adding environment variable: {env_var.name}")
    elif container.env: # If "env" is not in config and there are existing env vars, clear them
        container.env[:] = []
        print(f"  Clearing all environment variables.")


    if "maxRetries" in config:
        execution_template_obj.max_retries = config["maxRetries"]
        print(f"  Setting maxRetries: {execution_template_obj.max_retries}")

    if "timeoutSeconds" in config:
        # Timeout is specified in seconds and needs to be converted to a Duration object
        duration_obj = Duration(seconds=config["timeoutSeconds"])
        execution_template_obj.timeout = duration_obj
        print(f"  Setting timeoutSeconds: {config['timeoutSeconds']} seconds")

    # The job object is now fully configured (either newly created or updated existing one)

    try:
        if job_exists:
            # For updating, the job object needs to include the name.
            # We already fetched the job, so its name is correct.
            # The update_job method intelligently uses a field mask based on changes.
            operation = client.update_job(job=job_to_configure)
            print(f"Update operation started: {operation.operation.name}")
        else:
            operation = client.create_job(parent=parent, job=job_to_configure, job_id=job_name)
            print(f"Create operation started: {operation.operation.name}")

        # Wait for the operation to complete
        response = operation.result()
        print(f"Cloud Run job '{job_name}' successfully configured: {response.name}")
        return response

    except exceptions.GoogleAPICallError as e:
        print(f"Google API Error: {e.message}")
        print(f"Code: {e.code}")
        print(f"Details: {e.details}")
        print(f"Errors: {e.errors}")
        print(f"Request: {e.args[0].request_data}")
        print(f"Response: {e.args[0].response_data}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
