from pathlib import Path
import os
from typing import Union, Tuple
import subprocess
import time
import math


def do_chunking(input_video: Union[str, Path], out_dir: Union[str, Path], chunk_duration: int = 10) -> None:
    """Chunk a video into smaller segments using ffmpeg."""
    tag = Path(input_video).stem
    output_dir = os.path.join(out_dir, tag)
    os.makedirs(output_dir, exist_ok=True)

    # Get total duration of video (in seconds)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_video,
    ]
    duration = float(subprocess.check_output(cmd).decode().strip())
    time.sleep(1)  # Sleep to ensure ffprobe has time to process
    print(f"video duration={duration}")
    num_chunks = math.ceil(duration / chunk_duration)

    for i in range(num_chunks):
        start = i * chunk_duration
        end_time = (i + 1) * chunk_duration
        end_time = min(end_time, duration)
        actual_duration = min(chunk_duration, duration - start)
        print(f"chunk#{i}, {chunk_duration}, {start}, {actual_duration}, {end_time} == {start + actual_duration}")
        if actual_duration <= 0.0:
            break  # Nothing more to extract
        output_file = os.path.join(output_dir, f"{tag}_{i:03d}.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_video,
            "-ss",
            str(start),
            "-to",
            str(end_time),
            "-c:v",
            "libx264",  # Re-encode video
            "-preset",
            "veryfast",  # Encoding speed
            "-c:a",
            "aac",  # Re-encode audio
            "-strict",
            "experimental",  # Allow experimental AAC encoder
            output_file,
        ]
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts:
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                break  # Command succeeded, exit the loop

            print(f"Error in chunk {i} (attempt {attempts + 1}/{max_attempts}): {result.stderr.decode()}")
            attempts += 1
            time.sleep(1)  # Wait before retrying

        if attempts == max_attempts:
            print(f"Failed to process chunk {i} after {max_attempts} attempts")
        else:
            time.sleep(0.5)  # Only sleep on success
        time.sleep(1)  # Sleep to ensure ffmpeg has time to process

        # Check the duration of the output file
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            output_file,
        ]
        try:
            output_duration = float(subprocess.check_output(cmd).decode().strip())
        except Exception:
            output_duration = -1
        print(f"Output file {output_file} duration: {output_duration} seconds")

        # Break if the output file is empty or has no duration
        if output_duration <= 0:
            print(f"Output file {output_file} is empty or has no duration, stopping.")
            break
