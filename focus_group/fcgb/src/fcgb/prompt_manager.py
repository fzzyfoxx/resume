import os
import re
from typing import Dict, List, Optional
from importlib.resources import files

class PromptManager:
    def __init__(self, version_config: Dict[str, Optional[str]] = {}, path: Optional[str] = None):
        """
        Initialize the PromptManager.

        :param version_config: A dictionary specifying the version of each prompt to use.
        :param path: The directory to search for prompts. If None, defaults to the 'prompts' directory in the fcgb package.
        """
        self.version_config = version_config
        self.path = path or str(files('fcgb').joinpath('prompts'))
        self.prompts = self._explore_prompts()

    def _explore_prompts(self) -> Dict[str, List[Dict]]:
        """Explore the prompts directory and collect all prompt files."""
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
        """Load and return the content of a specific prompt."""
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
        """Return default prompts for a given list of prompt names."""
        result = {}
        for prompt_name in prompts:
            if prompt_name in self.prompts:
                result[prompt_name] = self.get_prompt(prompt_name)
            else:
                print(f"Warning: Prompt '{prompt_name}' not found.")
        return result

    def prompt_versions(self, prompt_name: str) -> None:
        """Print all versions of a prompt with the default version marked."""
        if prompt_name not in self.prompts:
            print(f"Prompt '{prompt_name}' not found.")
            return

        for prompt in self.prompts[prompt_name]:
            default_marker = " (default)" if prompt['default'] else ""
            print(f"Version {prompt['version']}{default_marker}")