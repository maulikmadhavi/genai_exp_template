import dspy
import re
from typing import List, Tuple

from dspy_video_test import Video


def parse_prompt_to_signature(
    prompt_text: str, class_name: str = "GeneratedSignature", docstring: str | None = None
) -> type:
    """Automatically generate a DSPy Signature class from structured prompt text."""

    def extract_fields(text: str, field_type: str) -> List[Tuple[str, str]]:
        """Extract field definitions from the prompt text."""
        pattern = r"Input fields?:(.*?)(?=Output fields?:|$)" if field_type == "input" else r"Output fields?:(.*?)$"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []

        section = match.group(1).strip()

        # Try numbered fields first, then simple format
        for pattern in [
            r"(\d+)\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^\n]+)",
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^\n]+)",
        ]:
            matches = re.findall(pattern, section, re.MULTILINE)
            if matches:
                return [(m[-2].strip(), m[-1].strip()) for m in matches if not re.match(r"\d+", m[-1].strip())]
        return []

    def is_video_field(name: str, desc: str) -> bool:
        """Check if field should be Video type."""
        return any(indicator in (name + desc).lower() for indicator in ["video", "mp4", "avi", "mov"])

    # Extract and process fields
    input_fields = extract_fields(prompt_text, "input")
    output_fields = extract_fields(prompt_text, "output")

    class_attrs = {}
    annotations = {}

    # Add docstring if provided
    if docstring:
        class_attrs["__doc__"] = docstring

    # Process all fields
    for field_name, description in input_fields:
        if is_video_field(field_name, description):
            annotations[field_name] = Video
        else:
            annotations[field_name] = str
        class_attrs[field_name] = dspy.InputField(desc=description)

    for field_name, description in output_fields:
        annotations[field_name] = str
        class_attrs[field_name] = dspy.OutputField(desc=description)

    class_attrs["__annotations__"] = annotations
    return type(class_name, (dspy.Signature,), class_attrs)


# Example usage
prompt = """
Input field:
video_in: video chunk data input
Output fields:
hands_and_safety: hands on wheel and safety description from the trainee/driver perspective
seatbelt: seatbelt description from the trainer and trainee perspective
phone_use: phone use description from trainer perspective
eye_close_doze: eye close and doze or tired or yawning description from trainee perspective
loose_items: loose items on the dashboard from scene perspective
distracted: distracted behaviour from trainer and trainee perspective
posture: posture description such as slouching or feet on dashboard from trainer and trainee perspective
safety: safe/unsafe in one word from any of the violation detected from the scene perspective
"""

# Generate the signature class automatically with proper docstring
detailed_instructions = """Output the video description. The person near to the camera is the trainer/instructor and the person far from the camera is the trainee/driver. Generate the violation indication only if you are sure about it, do not guess/hallucinate. If you are not sure about any of the fields, output 'Not sure'.
This are the user defined violation categories:
1. Trainer is away from the steering wheel and trainee is holding the steering wheel.
2. Hands on wheel and safety description from the trainee/driver perspective.
3. Seatbelt description from the trainer and trainee perspective.
4. Phone use description from trainer perspective.
5. Eye close and doze or tired or yawning description from trainee perspective.
6. Loose items on the dashboard from scene perspective."""

VideoSummary = parse_prompt_to_signature(prompt, "VideoSummary", detailed_instructions)


# Manual definition for comparison
class VideoSummaryManual(dspy.Signature):
    """Output the video description. The person near to the camera is the trainer/instructor and the person far from the camera is the trainee/driver. Generate the violation indication only if you are sure about it, do not guess/hallucinate. If you are not sure about any of the fields, output 'Not sure'.
    Trainer is away from the steering wheel and trainee is holding the steering wheel.
    """

    video_in: Video = dspy.InputField(desc="video chunk data input")
    hands_and_safety: str = dspy.OutputField(
        desc="hands on wheel and safety description from the trainee/driver perspective"
    )
    seatbelt: str = dspy.OutputField(desc="seatbelt description from the trainer and trainee perspective")
    phone_use: str = dspy.OutputField(desc="phone use description from trainer perspective")
    eye_close_doze: str = dspy.OutputField(
        desc="eye close and doze or tired or yawning description from trainee perspective"
    )
    loose_items: str = dspy.OutputField(desc="loose items on the dashboard from scene perspective")
    distracted: str = dspy.OutputField(desc="distracted behaviour from trainer and trainee perspective")
    posture: str = dspy.OutputField(
        desc="posture description such as slouching or feet on dashboard from trainer and trainee perspective"
    )
    safety: str = dspy.OutputField(
        desc="safe/unsafe in one word from any of the violation detected from the scene perspective"
    )


VideoSummaryManual.__doc__ = """
    Output the video description. The person near to the camera is the trainer/instructor and the person far from the camera is the trainee/driver. Generate the violation indication only if you are sure about it, do not guess/hallucinate. If you are not sure about any of the fields, output 'Not sure'.
    This are the user defined violation categories:
    1. Trainer is away from the steering wheel and trainee is holding the steering wheel.
    2. Hands on wheel and safety description from the trainee/driver perspective.
    3. Seatbelt description from the trainer and trainee perspective.
    4. Phone use description from trainer perspective.
    5. Eye close and doze or tired or yawning description from trainee perspective.
    6. Loose items on the dashboard from scene perspective.
"""

if __name__ == "__main__":
    print(f"Generated: {VideoSummary}")
    print(f"Manual: {VideoSummaryManual}")

    # Test the differences
    print("\n=== INSTRUCTIONS COMPARISON ===")
    print("Generated instructions:")
    print(repr(VideoSummary.instructions))
    print("\nManual instructions:")
    print(repr(VideoSummaryManual.instructions))

    # Test with LLM
    try:
        import litellm

        lm = dspy.LM(
            model="openai/Qwen/Qwen2.5-VL-7B-Instruct",
            api_key="fake-key",
            api_base="http://localhost:8005/v1",
            temperature=0.0,
            seed=42,
            max_tokens=1024,
        )
        dspy.configure(lm=lm)
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

        class CoTAnalyzer(dspy.Module):
            def __init__(self) -> None:
                self.predictor = dspy.Predict(VideoSummary)

            def __call__(self, **kwargs):
                return self.predictor(**kwargs)

        class CoTAnalyzerManual(dspy.Module):
            def __init__(self) -> None:
                self.predictor = dspy.Predict(VideoSummaryManual)

            def __call__(self, **kwargs):
                return self.predictor(**kwargs)

        module = CoTAnalyzer()
        module_manual = CoTAnalyzerManual()
        video_chunk_file = "/home/maulik/projects/VLM/vlm_transport/data/SAFEC/processed/8/8_021.mp4"

        video_in = Video(url=video_chunk_file)
        result = module(video_in=video_in)
        print("\nResult Generated:", result)

        result_manual = module_manual(video_in=video_in)
        print("Result Manual:", result_manual)

    except ImportError:
        print("LiteLL M not available, skipping inference test")
    except Exception as e:
        print(f"Error during inference: {e}")
