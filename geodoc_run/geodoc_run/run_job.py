import argparse
from geodoc_run.src.job import run_cloud_run_job

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

    # Call the function to run the Cloud Run job.
    success = run_cloud_run_job(args.job_name, parsed_env_vars)

    if success:
        print("\nCloud Run job execution request processed.")
        print("Check the Cloud Run console for the actual job execution details and logs.")
    else:
        print("\nFailed to process Cloud Run job execution request.")
        print("Please review the error messages above and your configuration.")

if __name__ == "__main__":
    main()