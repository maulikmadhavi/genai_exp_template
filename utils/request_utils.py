import requests
from prompts_lib.defaults import (
    BASE_VLM_PROMPT,
    BASE_LLM_PROMPT,
    LLM_MAX_LENGTH,
    VLM_MAX_LENGTH,
    LLM_TEMPERATURE,
    VLM_TEMPERATURE,
    SEED,
)


def send_llm_summary_request(**kwargs) -> str | None:
    """Send a text query prompt to the VLLM API."""
    headers = {
        "Content-Type": "application/json",
    }
    # Extract parameters from kwargs
    vllm_api_endpoint = kwargs.get("vllm_api_endpoint", "http://localhost:8000/v1/chat/completions")
    vllm_model = kwargs.get("vllm_model")
    prompt = kwargs.get("prompt", "")
    summary_report = kwargs.get("summary_report", "")
    system_prompt = kwargs.get("system_prompt", BASE_LLM_PROMPT)
    max_length = kwargs.get("max_length", LLM_MAX_LENGTH)
    temperature = kwargs.get("temperature", LLM_TEMPERATURE)

    # Test if no prompt is provided
    if not prompt and not summary_report:
        print("No prompt provided. Please provide a prompt or summary_report.")
        return False
    # Define the payload template
    payload = {
        "model": vllm_model,
        "temperature": temperature,
        "max_tokens": max_length,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt + "\n\n" + summary_report,
                    },
                ],
            },
        ],
        "seed": SEED,
    }

    try:
        response = requests.post(vllm_api_endpoint, headers=headers, json=payload)
        result = response.json()
        result = str(result["choices"][0]["message"]["content"])
        return result
    except Exception as e:
        print(f"LLM error -- send_text_query_prompt: {e}")
    return False


def send_vlm_video_request(**kwargs) -> str | bool:
    headers = {
        "Content-Type": "application/json",
    }
    # Extract parameters from kwargs
    vllm_api_endpoint = kwargs.get("vllm_api_endpoint", "http://localhost:8000/v1/chat/completions")
    vllm_model = kwargs.get("vllm_model")
    prompt = kwargs.get("prompt", "")
    video_base64 = kwargs.get("video_base64", "")
    system_prompt = kwargs.get("system_prompt", BASE_VLM_PROMPT)
    max_length = kwargs.get("max_length", VLM_MAX_LENGTH)
    temperature = kwargs.get("temperature", VLM_TEMPERATURE)
    if not prompt or not video_base64:
        print("No prompt or video_base64 provided. Please provide both.")
        return False

    # Define the payload template
    payload = {
        "model": vllm_model,
        "temperature": temperature,
        "max_tokens": max_length,
        "seed": SEED,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
    }

    try:
        response = requests.post(vllm_api_endpoint, headers=headers, json=payload)
        result = response.json()

        # tokens = result["usage"]["completion_tokens"]
        result = str(result["choices"][0]["message"]["content"])
        return result

    except Exception as e:
        print(f"vlm error: {e}")
        return False


def send_vlm_image_request(**kwargs) -> str | bool:
    headers = {
        "Content-Type": "application/json",
    }
    # Extract parameters from kwargs
    vllm_api_endpoint = kwargs.get("vllm_api_endpoint", "http://localhost:8000/v1/chat/completions")
    vllm_model = kwargs.get("vllm_model")
    if not vllm_model:
        print("No vllm_model provided. Please provide a model name.")
        return False
    prompt = kwargs.get("prompt", "")
    image_base64 = kwargs.get("image_base64", "")
    system_prompt = kwargs.get("system_prompt", BASE_VLM_PROMPT)
    max_length = kwargs.get("max_length", VLM_MAX_LENGTH)
    temperature = kwargs.get("temperature", VLM_TEMPERATURE)
    if not prompt or not image_base64:
        print("No prompt or image_base64 provided. Please provide both.")
        return False
    # Clean the base64 string
    clean_b64 = image_base64.replace("\n", "").strip()
    if len(clean_b64) % 4 != 0:
        clean_b64 += "=" * (4 - (len(clean_b64) % 4))

    payload = {
        "model": vllm_model,
        "temperature": temperature,
        "max_tokens": max_length,
        "seed": SEED,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"}},
                ],
            },
        ],
    }

    try:
        response = requests.post(vllm_api_endpoint, headers=headers, json=payload)
        result = response.json()
        result = str(result["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"vlm error: {e}")
        result = False

    return result
