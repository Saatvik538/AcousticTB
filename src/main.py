from data.feature_extraction import process_audio_files
from data.preprocess import update_metadata
import os
import pandas as pd

# Paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
# Load metadata

additional_vars = pd.read_csv("data/raw/solicited_coughs/metadata/CODA_TB_additional_variables_train.csv")
clinical_meta = pd.read_csv("data/raw/solicited_coughs/metadata/CODA_TB_Clinical_Meta_Info.csv")
solicited_meta = pd.read_csv("data/raw/solicited_coughs/metadata/CODA_TB_Solicited_Meta_Info.csv")
longitudinal_meta = pd.read_csv("data/raw/longitudinal_coughs/metadata/metadata.csv")

# Merge solicited metadata
solicited_full = (
    solicited_meta.merge(additional_vars, on="participant", how="left")
    .merge(clinical_meta, on="participant", how="left")
)

# Add labels (e.g., sound_prediction_score)
solicited_full["label"] = (solicited_full["sound_prediction_score"] > 0.8).astype(int)  # Binary label

# Process longitudinal metadata
longitudinal_meta["label"] = (longitudinal_meta["sound_prediction_score"] > 0.8).astype(int)

# Save processed metadata
solicited_full.to_csv("data/processed/solicited/merged_metadata.csv", index=False)
longitudinal_meta.to_csv("data/processed/longitudinal/processed_metadata.csv", index=False)

def main():
    # Solicited coughs
    solicited_audio_dir = os.path.join(RAW_DIR, "solicited_coughs/audio")
    solicited_metadata_dir = os.path.join(RAW_DIR, "solicited_coughs/metadata")
    solicited_feature_dir = os.path.join(PROCESSED_DIR, "solicited/features")

    process_audio_files(solicited_audio_dir, solicited_feature_dir)

    for metadata_file in os.listdir(solicited_metadata_dir):
        if metadata_file.endswith(".csv"):
            metadata_path = os.path.join(solicited_metadata_dir, metadata_file)
            update_metadata(metadata_path, solicited_audio_dir, solicited_feature_dir)

    # Longitudinal coughs (week1 and week2)
    longitudinal_audio_dir = os.path.join(RAW_DIR, "longitudinal_coughs/audio")
    longitudinal_metadata_dir = os.path.join(RAW_DIR, "longitudinal_coughs/metadata")
    for week in ["week1", "week2"]:
        week_audio_dir = os.path.join(longitudinal_audio_dir, week)
        week_feature_dir = os.path.join(PROCESSED_DIR, f"longitudinal/{week}_features")

        process_audio_files(week_audio_dir, week_feature_dir)

        for metadata_file in os.listdir(longitudinal_metadata_dir):
            if metadata_file.endswith(".csv") and week in metadata_file:
                metadata_path = os.path.join(longitudinal_metadata_dir, metadata_file)
                update_metadata(metadata_path, week_audio_dir, week_feature_dir)

if __name__ == "__main__":
    main()