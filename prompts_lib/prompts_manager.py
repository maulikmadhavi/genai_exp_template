import yaml
from pathlib import Path


class PromptsManager:
    prompt_file: str | Path
    prompts: dict

    def __init__(self, prompts_file: str | Path):
        self.prompts_file = prompts_file
        self.prompts = self.load_prompts()

    def load_prompts(self):
        with open(self.prompts_file, "r") as file:
            return yaml.safe_load(file)

    def get_prompt(self, prompt_name: str) -> str:
        return self.prompts.get(prompt_name, "Prompt not found.")
