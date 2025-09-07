# Introduction
This project is designed for custom prompt execution flow with LLMs and VLMs.
Current focus is to cover different aspects such as prompts handling, multi-modal inputs formatting for VLM, REST API calls for standard OPENAI compatible model serving, output manipulation and storing, evaluation and mlflow based experiment tracking.

# Overall Architecture
## Directory Structure
The project is organized with the following directory structure:

```
|-- main.py                 # Main entry point for the application
|-- configs/                # Configuration files for different components
|-- prompts_lib/            # Directory containing prompt templates and related files
|   |-- prompts_manager.py  # Manages prompt templates and their retrieval
|-- mm_input_lib/           # Directory for multi-modal input handling
|   |-- mm_input_formatter.py # Formats multi-modal inputs for VLMs
|-- output_lib/             # Directory for output processing and storage
|   |-- logging.py           # Manages output manipulation and storage
|   |-- metrics.py           # Computes evaluation metrics for model outputs
|-- utils/               # Directory for evaluation and experiment tracking
|   |-- request_utils.py     # Handles REST API requests to model serving endpoints
|   |-- video_processing_utils.py   # Utilities for processing video inputs
|-- executor_lib/               # Directory for execution flow management
|   |-- base.py            # Base class for executors
|-- exp/                    # Directory for experiments
|   |-- car_attributes/     # Example experiment for car attributes extraction
|       |-- run.py          # Script to run the car attributes experiment
|       |-- config.yaml    # Configuration file for the car attributes experiment
...
|-- README.md               # Project documentation
|-- pixi.toml              # Environment setup file