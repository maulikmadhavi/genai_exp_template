import pandas as pd
from pathlib import Path
from mm_inputs_lib.encoding_mm import (
    encode_base64_content_for_imagefile,
    process_video,
    encode_base64_content_from_file,
)
import os


def process_csv(
    csv_path: str, image_header_name: str = "image", selected_columns: list[str] = None, **kwargs
) -> pd.DataFrame:
    """Process a CSV file and return a DataFrame."""
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if image_header_name not in selected_columns:
        selected_columns = selected_columns + [image_header_name] if selected_columns else [image_header_name]

    # drop columns not in selected_columns
    if selected_columns:
        df = df[selected_columns]

    df["updated_image"] = df[image_header_name].apply(lambda x: str(Path(kwargs.get("image_dir", "")).joinpath(x)))
    # Process the image column
    if image_header_name in df.columns:
        df["image_base64"] = df["updated_image"].apply(encode_base64_content_for_imagefile)
    else:
        raise ValueError(f"Column '{image_header_name}' not found in CSV.")
    return df


def process_video_chunks_csv(
    csv_path: str, video_header_name: str = "video", selected_columns: list[str] = None, **kwargs
) -> pd.DataFrame:
    """Process a CSV file for video chunk input and return a DataFrame."""
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if video_header_name not in selected_columns:
        selected_columns = selected_columns + [video_header_name] if selected_columns else [video_header_name]

    # drop columns not in selected_columns
    if selected_columns:
        df = df[selected_columns]
    df["full_video_file"] = df[video_header_name].apply(lambda x: str(Path(kwargs.get("video_dir", "")).joinpath(x)))
    df["processed_file"] = df[video_header_name].apply(lambda x: str(Path(kwargs.get("process_dir", "")).joinpath(x)))

    # Debug: Print the paths to check
    print(f"Process dir: {kwargs.get('process_dir', '')}")
    print(f"Video dir: {kwargs.get('video_dir', '')}")
    for idx, row in df.head(3).iterrows():  # Check first 3 rows
        print(f"Row {idx}:")
        print(f"  Original video name: {row[video_header_name]}")
        print(f"  Full video file: {row['full_video_file']}")
        print(f"  Processed file: {row['processed_file']}")
        print(f"  Full video exists: {Path(row['full_video_file']).exists()}")
        print(f"  Processed file exists: {Path(row['processed_file']).exists()}")

    
    for _, x in df.iterrows():
        if not os.path.isdir(kwargs.get("process_dir", "")):
            os.makedirs(kwargs.get("process_dir", ""), exist_ok=True)
        if os.path.isfile(x["processed_file"]):  #  reduce redundant processing
            print(f"Processed file already exists, skipping: {x['processed_file']}")
            continue
        if not os.path.isfile(x["full_video_file"]):
            print(f"Input video file does not exist: {x['full_video_file']}")
            continue
        process_video(
            input_video_file=x["full_video_file"],
            output_video_file=x["processed_file"],
            total_samples=kwargs.get("total_samples", 10),
            fps=kwargs.get("process_fps", 1),
            resize=kwargs.get("resize", (640, 480)),
        )

    # Process the image column
    if video_header_name in df.columns:
        df["video_base64"] = df[df["processed_file"].apply(os.path.isfile)]["processed_file"].apply(encode_base64_content_from_file)
    else:
        raise ValueError(f"Column '{video_header_name}' not found in CSV.")
    return df
