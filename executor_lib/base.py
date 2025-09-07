import yaml


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


class BASE:
    def __init__(self, exp_config: str):
        self.exp_config = exp_config
        self.config = load_yaml(exp_config)

    def get_data_config(self) -> dict:
        return self.config.get("data", {})

    def get_prompt_config(self) -> dict:
        return self.config.get("prompt_template", {})

    def get_output_dir(self) -> str:
        return self.config.get("output_dir", "./results")

    def get_processor(self) -> str:
        return self.config.get("processor", None)

    def get_prompts_file_config(self) -> str:
        if not self.config.get("prompt_template", ""):
            raise ValueError("No prompt_template found in the configuration.")
        return load_yaml(self.config.get("prompt_template"))

    def run_execution(self):
        # Placeholder for execution logic
        pass
