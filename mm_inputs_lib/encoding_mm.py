import base64
import cv2
from decord import VideoReader, cpu
from typing import Union, Optional, List, Tuple
from pathlib import Path


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local video file to base64 format."""
    with open(file_path, "rb") as file:
        file_content = file.read()
        base64_encoded_content = base64.b64encode(file_content).decode("utf-8")
    return base64_encoded_content


def encode_base64_content_for_imagefile(image_path: str) -> str:
    """Encode a local image file to base64 format."""
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def process_video(
    input_video_file: str | Path,
    output_video_file: str | Path,
    total_samples: int = 10,
    fps: int = 1,
    resize: Tuple[int, int] = (640, 480),
) -> bool:
    try:
        if not Path(input_video_file).is_file():
            print(f"Input video file does not exist: {input_video_file}")
            return False
        Path(output_video_file).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(input_video_file, Path):  # If in Path format, convert to str
            input_video_file = str(input_video_file)
        vr = VideoReader(input_video_file, ctx=cpu(0))
        total_frame_num = len(vr)

        frame_height, frame_width, _ = vr[0].asnumpy().shape

        batch_array = []
        batch_array.extend(int(total_frame_num / total_samples * i) for i in range(total_samples))
        frames = vr.get_batch(batch_array).asnumpy()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
        if resize is None:
            out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

        else:
            out = cv2.VideoWriter(output_video_file, fourcc, fps, resize)
        for frame in frames:
            if resize is None:
                _frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(_frame)

            else:
                resize_frame = cv2.cvtColor(cv2.resize(frame, resize), cv2.COLOR_BGR2RGB)
                out.write(resize_frame)
        out.release()

    except cv2.error as e:
        print(f"resize: {resize} {input_video_file} {e}")

    return True
