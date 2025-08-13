import argparse
from geodoc_run.src.local_job import run_docker_container
from geodoc_config import get_service_config
import os 

def main():
    # Set up argument parsing for command-line inputs.
    parser = argparse.ArgumentParser(
        description="Run a Google Cloud Run job with custom environment variables."
    )
    parser.add_argument(
        "--job_name",
        required=True,
        help="The name of the Cloud Run job to execute."
    )
    parser.add_argument(
        "--e",
        action="append", # This allows specifying --env_var multiple times.
        default=[],
        help="Environment variable to pass to the job. Format: KEY=VALUE. "
             "Can be specified multiple times (e.g., --e FOO=bar --e BAZ=qux)."
    )

    args = parser.parse_args()

    # Parse the list of 'KEY=VALUE' strings into a dictionary.
    parsed_env_vars = {}
    for env_pair in args.e:
        if '=' in env_pair:
            key, value = env_pair.split('=', 1) # Split only on the first '=' to allow '=' in value.
            parsed_env_vars[key] = value
        else:
            print(f"Warning: Ignoring malformed environment variable argument '{env_pair}'. "
                  "Expected KEY=VALUE format. Please check your input.")
            
    # Add the Google Cloud project ID to the environment variables.
    general_gcp_config = get_service_config("gcp", "general")
    parsed_env_vars['GOOGLE_CLOUD_PROJECT'] = general_gcp_config['project_id']

    # Define the volume mapping for the Docker container.
    # This maps the local gcloud configuration directory to the container's gcloud config directory.
    volume_mapping = f"{os.path.expanduser('~')}/.config/gcloud:/root/.config/gcloud"

    # Call the function to run the container.
    result, container_id = run_docker_container(
        image_name=args.job_name,
        volume_mapping=volume_mapping,
        env_variables=parsed_env_vars
    )

    if result:
        print("\nDocker container launched successfully.")
        print(f"Container ID: {container_id}")
        print("You can view the logs using:")
        print(f"docker logs -f {container_id}")
    else:
        print("\nFailed to launch Docker container.")
        print("Please review the error messages above and your configuration.")

if __name__ == "__main__":
    main()