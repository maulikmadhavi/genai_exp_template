# Introduction
This project is designed for custom prompt execution flow with LLMs and VLMs.
Current focus is to cover different aspects such as prompts handling, multi-modal inputs formatting for VLM, REST API calls for standard OPENAI compatible model serving, output manipulation and storing, evaluation and mlflow based experiment tracking.

# Overall Architecture
|-- main.py : Main entry point for the application.
|-- configs/ : Configuration files for different components.
|-- prompts_lib/ : Directory containing prompt templates and related files.
    |--- prompts_manager.py : Manages prompt templates and their retrieval.
|--- mm_input_lib/ : Directory for multi-modal input handling.
    |--- mm_input_formatter.py : Formats multi-modal inputs for VLMs.
|--- api_lib/ : Directory for API interactions.
    |--- api_client.py : Handles REST API calls to LLM/VLM services.
|--- output_lib/ : Directory for output processing and storage.
    |--- output_handler.py : Manages output manipulation and storage.
|--- eval_lib/ : Directory for evaluation and experiment tracking.
    |--- eval_manager.py : Manages evaluation metrics and experiment tracking.  
    |--- mlflow_tracker.py : Integrates with MLflow for experiment tracking.
| --- exec_lib/ : Directory for execution flow management.
    |--- executor.py : Orchestrates the execution flow of prompts, inputs, and outputs.
