import pandas as pd
from pathlib import Path
from mm_inputs_lib.encoding_mm import encode_base64_content_for_imagefile


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
