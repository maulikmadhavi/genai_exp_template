import base64
import mimetypes
import os
import warnings
from typing import Any, Union
from urllib.parse import urlparse

import pydantic
import requests

from dspy.adapters.types.base_type import Type
import yaml
import dspy
import litellm


class Video(Type):
    url: str

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    def __init__(self, url: Any = None, *, download: bool = False, **data):
        """Create a Video.

        Parameters
        ----------
        url:
            The video source. Supported values include

            - ``str``: HTTP(S)/GS URL or local file path or data URI
            - ``bytes``: raw video bytes
            - ``dict`` with a single ``{"url": value}`` entry (legacy form)
            - already encoded data URI

        download:
            Whether remote URLs should be downloaded and encoded as base64.

        Any additional keyword arguments are passed to :class:`pydantic.BaseModel`.
        """

        if url is not None and "url" not in data:
            if isinstance(url, dict) and set(url.keys()) == {"url"}:
                data["url"] = url["url"]
            else:
                data["url"] = url

        if "url" in data:
            data["url"] = encode_video(data["url"], download_videos=download)

        super().__init__(**data)

    def format(self) -> list[dict[str, Any]] | str:
        try:
            video_url = encode_video(self.url)
        except Exception as e:
            raise ValueError(f"Failed to format video for DSPy: {e}")
        return [{"type": "video_url", "video_url": {"url": video_url}}]

    @classmethod
    def from_url(cls, url: str, download: bool = False):
        warnings.warn(
            "Video.from_url is deprecated; use Video(url) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(url, download=download)

    @classmethod
    def from_file(cls, file_path: str):
        warnings.warn(
            "Video.from_file is deprecated; use Video(file_path) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(file_path)

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        if "base64" in self.url:
            len_base64 = len(self.url.split("base64,")[1])
            video_type = self.url.split(";")[0].split("/")[-1]
            return f"Video(url=data:video/{video_type};base64,<VIDEO_BASE_64_ENCODED({len_base64!s})>)"
        return f"Video(url='{self.url}')"


def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme in ("http", "https", "gs"), result.netloc])
    except ValueError:
        return False


def encode_video(video: Union[str, bytes, dict], download_videos: bool = False) -> str:
    """
    Encode a video file or URL to a base64 data URI.

    Args:
        video: The video file or URL to encode. Can be file path, URL, bytes, or data URI.
        download_videos: Whether to download videos from URLs.

    Returns:
        str: The data URI of the video, or the URL if download_videos is False.
    """
    if isinstance(video, dict) and "url" in video:
        return video["url"]
    elif isinstance(video, str):
        if video.startswith("data:"):
            return video
        elif os.path.isfile(video):
            return encode_base64_content_from_file(video)
        elif is_url(video):
            if download_videos:
                return encode_video_from_url(video)
            else:
                return video
        else:
            raise ValueError(
                f"Unrecognized file string: {video}; If this file type should be supported, please open an issue."
            )
    elif isinstance(video, bytes):
        return encode_base64_content_from_bytes(video)
    elif isinstance(video, Video):
        return video.url
    else:
        print(f"Unsupported video type: {type(video)}")
        raise ValueError(f"Unsupported video type: {type(video)}")


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local video file to a base64 data URI."""
    with open(file_path, "rb") as file:
        file_content = file.read()
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "video/mp4"  # Default to mp4
    base64_encoded_content = base64.b64encode(file_content).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_content}"


def encode_base64_content_from_bytes(video_bytes: bytes) -> str:
    """Encode raw video bytes to a base64 data URI."""
    mime_type = "video/mp4"  # Default; could use file headers to guess
    base64_encoded_content = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_content}"


def encode_video_from_url(video_url: str) -> str:
    """Download and encode video from a URL to a base64 data URI."""
    response = requests.get(video_url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    mime_type = content_type if content_type else mimetypes.guess_type(video_url)[0] or "video/mp4"
    base64_encoded_content = base64.b64encode(response.content).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_content}"


def is_video(obj) -> bool:
    if isinstance(obj, str):
        if obj.startswith("data:"):
            return True
        elif os.path.isfile(obj):
            return True
        elif is_url(obj):
            return True
    elif isinstance(obj, bytes):
        return True
    return False


litellm._turn_on_debug()

lm = dspy.LM(
    model="openai/Qwen/Qwen2.5-VL-7B-Instruct",
    api_key="fake-key",
    api_base="http://localhost:8005/v1",
    temperature=0.0,
    seed=42,
    max_tokens=1024,
)
dspy.configure(
    lm=lm,
)
dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


class VideoDesc(dspy.Signature):
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
    # binary_safety: str = dspy.OutputField(desc="1/0 only based on safe/unsafe from any of the violation detected from the scene perspective")


VideoDesc.__doc__ = """
    Output the video description. The person near to the camera is the trainer/instructor and the person far from the camera is the trainee/driver. Generate the violation indication only if you are sure about it, do not guess/hallucinate. If you are not sure about any of the fields, output 'Not sure'.
    This are the user defined violation categories:
    1. Trainer is away from the steering wheel and trainee is holding the steering wheel.
    2. Hands on wheel and safety description from the trainee/driver perspective.
    3. Seatbelt description from the trainer and trainee perspective.
    4. Phone use description from trainer perspective.
    5. Eye close and doze or tired or yawning description from trainee perspective.
    6. Loose items on the dashboard from scene perspective.
"""


class CoTAnalyzer(dspy.Module):
    def __init__(self) -> None:
        self.predictor = dspy.Predict(VideoDesc)

    def __call__(self, **kwargs):
        return self.predictor(**kwargs)


# Consolidation Signature
class ConsolidatedVideoReport(dspy.Signature):
    """
    Consolidates chunk-level predictions into a single video-level summary.
    """

    chunk_predictions: list = dspy.InputField(
        desc="List of Prediction objects or dictionaries summarizing each video chunk"
    )
    summary: str = dspy.OutputField(desc="A concise report summarizing key safety observations from all chunks")


def binary_list_to_intervals(binary_list, chunk_dur: int = 10):
    intervals = []
    i = 0
    while i < len(binary_list):
        if binary_list[i] == 1:
            start = i
            while i + 1 < len(binary_list) and binary_list[i + 1] == 1:
                i += 1
            end = i
            intervals.append((start * chunk_dur, (end + 1) * chunk_dur))
        i += 1
    return intervals


# Define DSPy program for consolidation
class VideoReportConsolidator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.Predict(ConsolidatedVideoReport)

    def forward(self, chunk_predictions):
        # Convert Prediction objects to dicts if needed
        serialized_chunks = [p.dict() if hasattr(p, "dict") else p for p in chunk_predictions]
        unsafe = [1 if p.get("safety").lower() else 0 for ix, p in enumerate(chunk_predictions)]

        intervals = binary_list_to_intervals(unsafe)
        return self.summarizer(chunk_predictions=serialized_chunks), intervals


if __name__ == "__main__":
    # Save resullts into yaml file
    r_dict = {}
    for v in [1, 4, 5, 8, 12, 14, 18, 23, 25, 26, 28, 29]:
        chunk_outputs = []
        for i in range(1, 31):
            if v == 29 and i > 17:
                break
            video_chunk_file = f"/home/maulik/projects/VLM/vlm_transport/data/SAFEC/processed/{v}/{v}_{i:03d}.mp4"
            video_in = Video(url=video_chunk_file)
            print("Video input prepared for chunk:", video_chunk_file)

            module = CoTAnalyzer()
            result = module(video_in=video_in)
            print("Result:", result)
            r_dict[f"{v}_{i:03d}"] = result["safety"]
            chunk_outputs.append(result)
        # Suppose you already have a list of per-chunk Prediction objects

        # Initialize DSPy
        report_consolidator = VideoReportConsolidator()

        # Run consolidation
        final_report, intervals = report_consolidator(chunk_predictions=chunk_outputs)

        # Access the final summary
        print(final_report.summary)

    with open("dspy_video_results.yaml", "w") as f:
        yaml.dump(r_dict, f)
