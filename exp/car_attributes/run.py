import sys
import mlflow
from mlflow import MlflowClient
from PIL import Image
from pathlib import Path
import time

# Setup
exp_name = "car_attributes_experiment"
mlflow.set_experiment(exp_name)
client = MlflowClient()

sys.path.append(".")
from executor_lib.base import BASE
from mm_inputs_lib.data_processor import process_csv
from utils.request_utils import send_vlm_image_request

# Initialize
base_ = BASE("exp/car_attributes/make_type.yaml")
base_.run_execution()

# Get prompts and data
prompts = base_.get_prompts_file_config()
system_prompt = prompts.get("VLM_PROMPT")["SYSTEM"]
user_prompt = prompts.get("VLM_PROMPT")["USER"]
data = base_.get_data_config()
df = process_csv(**data)

# Single run with properly aligned logging
with mlflow.start_run(run_name="runs_1") as run:
    # Log prompts once
    mlflow.log_param("system_prompt", system_prompt)
    mlflow.log_param("user_prompt", user_prompt)

    run_id = run.info.run_id

    # Process each sample with step-based logging
    for step, row in df.iterrows():
        # Get VLM response
        input_reqs = dict(
            vllm_api_endpoint="http://localhost:8000/v1/chat/completions",
            vllm_model="qwen2.5-vl-3b-instruct",
            system_prompt=system_prompt,
            prompt=user_prompt,
            image_base64=row["image_base64"],
        )
        response = send_vlm_image_request(**input_reqs)

        # Get current timestamp for alignment
        timestamp = int(time.time() * 1000)  # MLflow uses milliseconds

        # Log image with step
        image_path = Path(row["updated_image"])
        if image_path.exists():
            image = Image.open(image_path)
            mlflow.log_image(image, key="input_image", step=step, timestamp=timestamp)

        # Log response using client API for step alignment
        # This creates a "response" metric that shows in Model metrics with steps
        client.log_text(run_id=run_id, text=response, artifact_file=f"responses/step_{step:03d}.txt")

        # Log response metadata as metrics with steps for visualization
        mlflow.log_metric("response_length", len(response), step=step, timestamp=timestamp)
        mlflow.log_metric("response_word_count", len(response.split()), step=step, timestamp=timestamp)
        mlflow.log_metric("response", response, step=step, timestamp=timestamp)

        # Create a response preview metric (first 50 chars)
        response_preview = response[:50].replace("\n", " ").replace("\t", " ")
        # Since we can't log text as metric, we'll use a workaround
        # Log the hash of the response so we can track uniqueness
        response_hash = hash(response) % 10000  # Keep it small for visualization
        mlflow.log_metric("response_hash", response_hash, step=step, timestamp=timestamp)

        print(f"Step {step}: Logged image and response")

print("Experiment completed!")
print("View results:")
print("- Images: Model metrics > input_image")
print("- Responses: Artifacts > responses/")
print("- Response metrics: Model metrics > response_length, response_word_count")
