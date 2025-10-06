import yaml
import os
from utils.general import load_yaml


class BASE:
    def __init__(self, exp_dir: str, exp_file: str):
        self.exp_config = os.path.abspath(os.path.join(exp_dir, exp_file))
        self.config = load_yaml(self.exp_config)

    def get_data_config(self) -> dict:
        return self.config.get("data", {})

    def get_prompt_config(self) -> dict:
        return self.config.get("prompt_template", {})

    def get_processor(self) -> str:
        return self.config.get("processor", None)

    def get_prompts_file_config(self) -> str:
        prompt_file = self.config.get("prompt_template", "")
        if not prompt_file:
            raise ValueError("No prompt_template found in the configuration.")
        return load_yaml(os.path.abspath(prompt_file))

    def get_experiment_name(self) -> str:
        return self.config["output"]["exp_name"]

    def get_output_dir(self) -> dict:
        return self.config.get("output", {}).get("output_dir", "./outputs")

    def run_execution(self):
        # Placeholder for execution logic
        pass