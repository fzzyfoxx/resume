import subprocess
import shlex
import os

def run_docker_container(
    image_name: str,
    volume_mapping: str = None,
    env_variables: dict = None
) -> tuple[bool, str | None]:
    """
    Runs a local Docker image in a detached, interactive mode.

    The container is launched in the background, allowing tqdm progress bars
    within the container's logs to display dynamically. This function does not
    block the calling terminal.

    Args:
        image_name (str): The name of the Docker image to run.
        volume_mapping (str, optional): A volume mapping in the format
                                        "host_path:container_path". Defaults to None.
        env_variables (dict, optional): A dictionary of environment variables
                                        to set in the container (e.g., {"KEY": "VALUE"}).
                                        Defaults to None.

    Returns:
        tuple[bool, str | None]: A tuple where the first element is True if the
                                 container was successfully launched, False otherwise.
                                 The second element is the container ID if successful,
                                 or None if an error occurred.
    """
    # Base Docker command for running in detached (-d) and interactive/TTY (-it) mode.
    # -it is crucial for tqdm's dynamic output to render correctly in logs.
    # -d ensures the 'docker run' command itself doesn't block the terminal.
    cmd = ["docker", "run", "-d", "-it"]

    # Add volume mapping if provided
    if volume_mapping:
        cmd.extend(["-v", volume_mapping])

    # Add environment variables if provided
    if env_variables:
        for key, value in env_variables.items():
            cmd.extend(["-e", f"{key}={shlex.quote(str(value))}"]) # shlex.quote handles spaces/special chars

    # Add the image name as the final argument
    cmd.append(image_name)

    try:
        # Execute the command. `capture_output=True` gets stdout/stderr,
        # `text=True` decodes output as text, `check=True` raises an exception
        # if the command returns a non-zero exit code.
        print(f"Attempting to launch Docker container with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # The `docker run -d` command outputs the container ID to stdout upon success.
        container_id = result.stdout.strip()

        print(f"\nSuccessfully launched Docker container '{image_name}' (ID: {container_id}).")
        return True, container_id
    except subprocess.CalledProcessError as e:
        # Handle errors if the docker command fails
        print(f"\nError launching Docker container '{image_name}':")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Exit Code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False, None
    except FileNotFoundError:
        # Handle cases where the 'docker' command itself is not found
        print("\nError: 'docker' command not found. Please ensure Docker Desktop is installed and configured correctly, and 'docker' is in your system's PATH.")
        return False, None
    except Exception as e:
        # Catch any other unexpected errors
        print(f"\nAn unexpected error occurred: {e}")
        return False, None
