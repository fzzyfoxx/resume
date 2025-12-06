## geodoc_config package

### Overview
The `geodoc_config` consists of a set of json configuration files that define various settings and parameters for the PerigonScout system and tools to access them.

#### Loading Tools
Functions defined in [loader.py](./geodoc_config/src/loader.py) provide easy access to load configuration files as Python dictionaries.
- `show_config_files() -> None`: Prints the list of available configuration files included in `configs map`.
- `load_config(config_name: str) -> dict`: Loads a configuration file by name defined in `configs map` and returns its contents as a dictionary.
- `get_service_config(service_name: str, key: str) -> dict`: Retrieves specific configuration settings for a given service e.g. 'worker', 'setup', 'service' for 'parcels' service.
- `load_config_by_path(folder: str, file: str) -> dict`: Loads a configuration file from a specified folder and file name even if it's not included in `configs map`.

#### Mappings
There are two mapping files that define relationships between services, sources, and configuration files:
- `configs_map.json`: Maps specific files with their names so that providing full paths is not necessary. [See file](./geodoc_config/configs/mapping/configs_map.json)
- `service2config.json`: Maps services to their respective configuration files for specific purposes. [See file](./geodoc_config/configs/mapping/service2config.json)

#### Configuration Files
The package includes various configuration files for different services and purposes. They are grouped into folders based on their functionality.
They are located in the [configs](./geodoc_config/configs) folder.

---
**Warning !** <br>
*Be aware that this package is not fully available for public access within this repository.*