import shutil
import tempfile
import docker
from pathlib import Path
from importlib.resources import files as importlib_files, as_file

def build_image(service_name: str):
    """
    Deploys a specified service by building its Docker image.
    Assumes .whl files are in geodoc_deploy/packages and
    requirements.txt is in each service's folder.
    """
    print(f"Attempting to deploy service: {service_name}")

    client = docker.from_env()

    # --- Step 1: Locate the service's files within the geodoc_deploy package ---
    # importlib.resources.files('geodoc_deploy') gives a Traversable object
    # representing the root of the installed geodoc_deploy package.
    # We then use .joinpath() to navigate to specific resources.
    
    # Path to the service's directory (e.g., geodoc_deploy/services/service_a/)
    service_package_path = importlib_files('geodoc_deploy').joinpath(f"services/{service_name}")

    if not service_package_path.is_dir(): # Check if the directory exists
        print(f"Error: Service directory '{service_name}' not found within geodoc_deploy package data.")
        return

    dockerfile_resource = service_package_path.joinpath("dockerfile")
    if not dockerfile_resource.is_file(): # Check if Dockerfile exists
        print(f"Error: Dockerfile not found for service '{service_name}'. Expected at {dockerfile_resource}")
        return

    requirements_resource = service_package_path.joinpath("requirements.txt")
    if not requirements_resource.is_file(): # Check if requirements.txt exists
        print(f"Error: requirements.txt not found for service '{service_name}'. Expected at {requirements_resource}")
        return
    
    script_resource = service_package_path.joinpath(f"{service_name}.py")
    if not script_resource.is_file(): # Check if the service script exists
        print(f"Error: Service script '{service_name}.py' not found in service directory '{service_name}'. Expected at {script_resource}")
        return

    # Path to the shared .whl files within the geodoc_deploy package
    whl_package_path = importlib_files('geodoc_deploy').joinpath("packages")
    if not whl_package_path.is_dir():
        print("Warning: No 'packages' directory found within geodoc_deploy package data. Ensure geodoc_loader.whl is bundled.")


    # --- Step 2: Prepare a temporary build context ---
    # Docker's build() method requires a local filesystem path for the context.
    # Since package data might be in a zip file (if installed as a wheel),
    # we need to extract everything to a temporary directory.
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_path = Path(temp_dir_str)
        print(f"Created temporary Docker build context: {temp_path}")

        # Copy service's Dockerfile and other service files (e.g., your_service_script.py, requirements.txt)
        # Using as_file() to get a filesystem path for the Traversable directory
        with as_file(service_package_path) as src_dir_fs_path:
            shutil.copytree(src_dir_fs_path, temp_path, dirs_exist_ok=True)
            print(f"Copied service '{service_name}' files to build context.")

        # Copy the shared .whl files to the 'packages' directory within the temporary context
        # (The Dockerfile expects them there)
        with as_file(whl_package_path) as whl_src_fs_path:
            target_wheels_dir = temp_path / "packages"
            shutil.copytree(whl_src_fs_path, target_wheels_dir)
            print(f"Copied internal wheels to build context: {target_wheels_dir}")


        # --- Step 3: Build the Docker Image ---
        try:
            image_tag = f"{service_name}:latest"
            print(f"Building Docker image: {image_tag} from context: {temp_path}")

            # The 'path' argument is the build context (temp_path),
            # 'dockerfile' is the Dockerfile's name relative to the context.
            image, build_logs = client.images.build(
                path=str(temp_path),
                dockerfile="dockerfile", # Dockerfile is now at the root of the temp context
                tag=image_tag,
                rm=True # Remove intermediate containers
            )

            for line in build_logs:
                if 'stream' in line:
                    print(line['stream'], end='')
                elif 'error' in line:
                    print(f"Build Error: {line['error']}", end='')
            print(f"Successfully built image: {image.tags}")

            # --- Step 4: (Optional) Run the Docker Container ---
            # You can add logic here to run the container, push to registry, etc.
            # For example:
            # print(f"Running container from image: {image_tag}")
            # container = client.containers.run(image_tag, detach=True, ports={'8000/tcp': 8000})
            # print(f"Container ID: {container.id}")
            # print(container.logs().decode('utf-8'))

        except docker.errors.BuildError as e:
            print(f"Docker build error: {e}")
            for line in e.build_log:
                if 'stream' in line:
                    print(line['stream'], end='')
        except docker.errors.APIError as e:
            print(f"Docker API error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
