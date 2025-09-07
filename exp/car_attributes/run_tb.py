import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

# Setup TensorBoard logging with PyTorch
exp_name = "car_attributes_experiment"
log_dir = f"tensorboard_logs/{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir)

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

print(f"Starting TensorBoard logging to: {log_dir}")
print(f"View results: tensorboard --logdir=tensorboard_logs")
print(f"Then open: http://localhost:6006")

# Log experiment metadata once
writer.add_text("experiment/system_prompt", system_prompt, global_step=0)
writer.add_text("experiment/user_prompt", user_prompt, global_step=0)

experiment_info = f"""
Experiment: {exp_name}
Model: qwen2.5-vl-3b-instruct
Total samples: {len(df)}
Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
writer.add_text("experiment/info", experiment_info, global_step=0)

# Process each sample with step-based logging
for step, row in df.iterrows():
    print(f"Processing step {step}...")

    # Get VLM response
    input_reqs = dict(
        vllm_api_endpoint="http://localhost:8000/v1/chat/completions",
        vllm_model="qwen2.5-vl-3b-instruct",
        system_prompt=system_prompt,
        prompt=user_prompt,
        image_base64=row["image_base64"],
    )
    response = send_vlm_image_request(**input_reqs)

    # Log image with step
    image_path = Path(row["updated_image"])
    if image_path.exists():
        # Load and prepare image for TensorBoard
        image = Image.open(image_path)

        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Convert to PyTorch tensor format: (C, H, W)
        if len(img_array.shape) == 3:  # Color image (H, W, C)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C, H, W)
        else:  # Grayscale (H, W)
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)

        # Normalize to [0, 1] if needed
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float() / 255.0

        # Log image - this will show in Images tab with step navigation!
        writer.add_image("input_images", img_tensor, global_step=step)

        # Log image metadata as scalars
        writer.add_scalar("image_metadata/width", image.size[0], global_step=step)
        writer.add_scalar("image_metadata/height", image.size[1], global_step=step)
        # Set title of figure as response
        # writer.add_figure()
        # writer.add_text("responses", response, global_step=step)
    # Log response as text - this will show in Text tab with step navigation!
    summary_text = f"""
        **Step {step} Summary:**
        - Image: {image_path.name}
        - Image size: {image.size[0]}x{image.size[1]}
        - Response length: {len(response)} chars
        - Word count: {len(response.split())} words

        **Response:**
        {response}
    """
    writer.add_text("responses", summary_text, global_step=step)

    # Log response metrics as scalars
    writer.add_scalar("response_metrics/length", len(response), global_step=step)
    writer.add_scalar("response_metrics/word_count", len(response.split()), global_step=step)
    writer.add_scalar("response_metrics/sentence_count", response.count("."), global_step=step)

    # Optional: Log response statistics
    unique_words = len(set(response.lower().split()))
    writer.add_scalar("response_metrics/unique_words", unique_words, global_step=step)

    # Force write to disk so you can view immediately
    writer.flush()

    print(f"✅ Step {step}: Logged image and response to TensorBoard")

# Log experiment completion
completion_info = f"""
Experiment completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total samples processed: {len(df)}
"""
writer.add_text("experiment/completion", completion_info, global_step=len(df))

# Close the writer
writer.close()

print(f"\n🎉 Experiment completed!")
print(f"📊 View results in TensorBoard:")
print(f"   1. Run: tensorboard --logdir=tensorboard_logs")
print(f"   2. Open: http://localhost:6006")
print(f"📁 Log directory: {log_dir}")
