import json
from importlib.resources import files

def load_json(path):
    """Load a JSON file"""
    with path.open("r") as f:
        return json.load(f)
    
def load_config_by_path(folder, file):
    """
    Load a configuration file from a specified folder and file name.
    Args:
        folder (str): The folder where the configuration file is located.
        file (str): The name of the configuration file.
    Returns:
        dict: The loaded configuration data.
    """
    # Access the specified configuration file within the geodoc_config package
    config_path = files(f"geodoc_config.configs.{folder}").joinpath(file)
    
    # Load and return the JSON data
    return load_json(config_path)
    
def load_mapping():
    """
    Load the mapping configuration from the package's resources.
    Returns:
        dict: The mapping configuration loaded from the JSON file.
    """
    # Access the configs_map.json file within the geodoc_config package
    config_path = files("geodoc_config.configs.mapping").joinpath("configs_map.json")
    
    # Load and return the JSON data
    return load_json(config_path)

def show_config_files():
    """Display available configuration files from the mapping."""
    mapping = load_mapping()
    print("Available configuration files:")
    for key, value in mapping.items():
        print(f"{key}: {value.get('description', 'No description available')}")

def get_config_file_path(config_name):
    """
    Get the path to a specific configuration file.
    Args:
        config_name (str): The name of the configuration file.
    Returns:
        str: The path to the configuration file, or None if not found.
    """
    mapping = load_mapping()
    if config_name in mapping:
        return files('.'.join(["geodoc_config.configs", mapping[config_name]["folder"]])).joinpath(mapping[config_name]["file"])
    else:
        print(f"Configuration '{config_name}' not found.")
        return None
    
def load_config(config_name):
    """
    Load a specific configuration file.
    Args:
        config_name (str): The name of the configuration file.
    Returns:
        dict: The loaded configuration data, or None if not found.
    """
    config_path = get_config_file_path(config_name)
    if config_path:
        return load_json(config_path)
    return None

def load_service_mapping():
    """
    Load the service mapping configuration.
    Returns:
        dict: The service mapping configuration loaded from the JSON file.
    """
    # Access the service2config.json file within the geodoc_config package
    config_path = files("geodoc_config.configs.mapping").joinpath("service2config.json")
    
    # Load and return the JSON data
    return load_json(config_path)

def get_service_config(service_name, key):

    service_mapping = load_service_mapping()
    if service_name in service_mapping:
        service_config = service_mapping[service_name]
        if key in service_config:
            return load_config(service_config[key])
        else:
            print(f"Key '{key}' not found in service '{service_name}'.")
            return None
    else:
        print(f"Service '{service_name}' not found in service mapping.")
        return None