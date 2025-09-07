import yaml
import os
import logging


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def get_defaults() -> dict:
    return load_yaml("prompts_lib/vault/default.yaml")


def set_up_exp_dir(exp_dir: str) -> None:
    """Set up experiment directory. If it exists, create a new one with an incremented suffix."""
    print(f"Setting up experiment directory at: {exp_dir}")
    if os.path.exists(exp_dir):
        # create new +1 directory [runs=> runs1, runs1=> runs2, ...]
        base_dir = exp_dir.rstrip("/").rstrip("\\")
        version = 1
        new_dir = f"{base_dir}_{version}"
        while os.path.exists(new_dir):
            version += 1
            new_dir = f"{base_dir}_{version}"
        os.makedirs(new_dir)
        print(f"Created new experiment directory: {new_dir}")
        return new_dir

    os.makedirs(exp_dir)
    return exp_dir


def setup_logger(log_file: str):
    """Simplified logger setup with no duplicates"""
    # Clear any existing handlers
    logging.getLogger().handlers.clear()

    # Configure logging with both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    # Reduce urllib3 verbosity
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logging.getLogger(__name__)
