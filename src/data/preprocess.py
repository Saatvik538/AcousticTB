import pandas as pd
import os

def update_metadata(csv_path, audio_dir, feature_dir):
    """
    Update metadata CSV by appending paths to corresponding feature files.

    Args:
        csv_path (str): Path to the metadata CSV.
        audio_dir (str): Directory containing the audio files.
        feature_dir (str): Directory containing extracted feature files.
    """
    df = pd.read_csv(csv_path)

    updated_data = []
    for _, row in df.iterrows():
        audio_file = os.path.join(audio_dir, row["filename"])
        feature_file = os.path.join(feature_dir, f"{os.path.splitext(row['filename'])[0]}_features.npy")

        if os.path.exists(feature_file):
            updated_data.append({**row, "feature_path": feature_file})
        else:
            print(f"Feature file missing for: {audio_file}")

    updated_df = pd.DataFrame(updated_data)
    updated_csv_path = os.path.join(feature_dir, "updated_metadata.csv")
    updated_df.to_csv(updated_csv_path, index=False)
    print(f"Updated metadata saved to: {updated_csv_path}")