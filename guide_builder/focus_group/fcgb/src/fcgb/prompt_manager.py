import os
import re
from typing import Dict, List, Optional
from importlib.resources import files

class PromptManager:
    """
    A class for managing and retrieving prompts from a specified directory.

    This class allows for the organization of prompts by name and version, enabling the retrieval
    of specific versions or the default version of a prompt. It is designed to work with text files
    stored in a directory, where each file represents a prompt.

    Attributes:
        version_config: A dictionary specifying the version of each prompt to use.
        path: The directory to search for prompts.
        prompts: A dictionary containing all discovered prompts, organized by name and version.
    """

    def __init__(self, version_config: Dict[str, Optional[str]] = {}, path: Optional[str] = None):
        """
        Initialize the PromptManager.

        Args:
            version_config: A dictionary specifying the version of each prompt to use.
            path: The directory to search for prompts. If None, defaults to the 'prompts' directory in the fcgb package.
        """
        self.version_config = version_config
        self.path = path or str(files('fcgb').joinpath('prompts'))
        self.prompts = self._explore_prompts()

    def _explore_prompts(self) -> Dict[str, List[Dict]]:
        """
        Explore the prompts directory and collect all prompt files.

        This method scans the specified directory for text files representing prompts. It organizes
        the prompts by name and version, and marks the default version based on the version_config
        or the latest version.

        Returns:
            Dict[str, List[Dict]]: A dictionary where each key is a prompt name, and the value is a list
            of dictionaries containing the path, version, and default status of each prompt.
        """
        prompts = {}
        version_pattern = re.compile(r'_v(\d+)\.(\d+)\.txt$')

        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith('.txt'):
                    match = version_pattern.search(file)
                    if match:
                        name = file[:match.start()]
                        version = float(f"{match.group(1)}.{match.group(2)}")
                        prompt_path = os.path.join(root, file)

                        if name not in prompts:
                            prompts[name] = []

                        prompts[name].append({'path': prompt_path, 'version': version, 'default': False})

        # Set default versions based on version_config or latest
        for name, prompt_list in prompts.items():
            prompt_list.sort(key=lambda x: x['version'], reverse=True)
            default_version = self.version_config.get(name, 'latest')
            if default_version is None or default_version == 'latest':
                prompt_list[0]['default'] = True
            else:
                for prompt in prompt_list:
                    if prompt['version'] == float(default_version):
                        prompt['default'] = True
                        break
                else:
                    print(f"Warning: Specified version {default_version} for prompt '{name}' not found. Using latest.")
                    prompt_list[0]['default'] = True

        # Warn about missing prompts in version_config
        for name in self.version_config.keys():
            if name not in prompts:
                print(f"Warning: Prompt '{name}' not found in the prompts directory.")

        return prompts

    def get_prompt(self, prompt_name: str, version: Optional[str] = None) -> str:
        """
        Load and return the content of a specific prompt.

        Args:
            prompt_name: The name of the prompt to retrieve.
            version: The version of the prompt to retrieve. If None, the default version is used.

        Returns:
            str: The content of the specified prompt.

        Raises:
            ValueError: If the prompt or the specified version is not found.
        """
        if prompt_name:
            if prompt_name not in self.prompts:
                raise ValueError(f"Prompt '{prompt_name}' not found.")

            for prompt in self.prompts[prompt_name]:
                if version is None and prompt['default']:
                    version = prompt['version']
                if version is not None:
                    if prompt['version'] == float(version):
                        with open(prompt['path'], 'r') as file:
                            return file.read()

            raise ValueError(f"Version {version} for prompt '{prompt_name}' not found.")
        return None

    def get_prompts(self, prompts: List[str]) -> Dict[str, str]:
        """
        Return default prompts for a given list of prompt names.

        Args:
            prompts: A list of prompt names to retrieve.

        Returns:
            Dict[str, str]: A dictionary where each key is a prompt name, and the value is the content of the default prompt.
        """
        result = {}
        for prompt_name in prompts:
            if prompt_name in self.prompts:
                result[prompt_name] = self.get_prompt(prompt_name)
            else:
                print(f"Warning: Prompt '{prompt_name}' not found.")
        return result

    def prompt_versions(self, prompt_name: str) -> None:
        """
        Print all versions of a prompt with the default version marked.

        Args:
            prompt_name: The name of the prompt to list versions for.
        """
        if prompt_name not in self.prompts:
            print(f"Prompt '{prompt_name}' not found.")
            return

        for prompt in self.prompts[prompt_name]:
            default_marker = " (default)" if prompt['default'] else ""
            print(f"Version {prompt['version']}{default_marker}")