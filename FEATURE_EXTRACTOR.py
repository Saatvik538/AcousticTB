import os
import tempfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# Set a custom temp directory
temp_dir = os.path.join(os.getcwd(), "temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir

tempfile.tempdir = temp_dir

# Paths
RAW_DIR = os.path.join("data", "raw")
SOLICITED_META_DIR = os.path.join(RAW_DIR, "solicited_coughs", "metadata")
PROCESSED_DIR = os.path.join("data", "processed")

SOLICITED_META_FILE = os.path.join(SOLICITED_META_DIR, "CODA_TB_Solicited_Meta_Info.csv")
SOLICITED_META_FILE = os.path.normpath(SOLICITED_META_FILE)

# Load VGGish model
VGGISH_URL = "https://tfhub.dev/google/vggish/1"
model_dir = 'C:/Users/Saatvik/.vscode/AcousticTB/'
vggish_model = tf.saved_model.load(model_dir)
def extract_features(audio_path, sample_rate=16000):
    # Load audio as mono and resample if necessary
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Optionally, you might want to perform further preprocessing like:
    # - Convert the waveform to a Mel-spectrogram, MFCCs, or another feature representation
    # - Normalize or standardize the audio data if needed

    # For VGGish, you'll need a specific feature extraction method, typically a mel-spectrogram or log-mel spectrogram
    # Example feature extraction: convert audio to a Mel spectrogram (shape (time, features))
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmax=8000)

    # Transpose the spectrogram to the required format (for VGGish)
    mel_spectrogram = mel_spectrogram.T

    # Make sure mel_spectrogram has shape (None, 64), e.g., padding or trimming if necessary
    if mel_spectrogram.shape[1] != 64:
        mel_spectrogram = mel_spectrogram[:, :64]

    return mel_spectrogram
# def extract_features(audio_path, sample_rate=16000):
#     try:
#         # Load audio file using librosa
#         audio, sr = librosa.load(audio_path, sr=sample_rate)
#         audio = librosa.util.normalize(audio)

#         # Convert audio to a 1D tensor
#         audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

#         # Pass the 1D tensor to the model (no reshaping needed)
#         embeddings = vggish_model(audio_tensor)
#         return embeddings.numpy().flatten()
#     except Exception as e:
#         print(f"Error processing {audio_path}: {e}")
#         return None

def process_metadata(metadata_file, audio_dir):
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata = pd.read_csv(metadata_file)
    metadata["feature_path"] = None

    output_dir = os.path.join(PROCESSED_DIR, "features")
    os.makedirs(output_dir, exist_ok=True)

    # Process only 5 files for testing
    for idx, row in tqdm(metadata.iterrows(), total=min(len(metadata), 5)):
        filename = row["filename"]
        audio_path = os.path.join(audio_dir, filename)

        if os.path.exists(audio_path):
            features = extract_features(audio_path)
            if features is not None:
                feature_file = os.path.join(output_dir, f"{filename}.npy")
                np.save(feature_file, features)
                metadata.at[idx, "feature_path"] = feature_file
        else:
            print(f"File not found: {audio_path}")

    return metadata

def main():
    solicited_audio_dir = os.path.join(RAW_DIR, "solicited_coughs", "audio")
    if not os.path.exists(solicited_audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {solicited_audio_dir}")

    solicited_metadata = process_metadata(SOLICITED_META_FILE, solicited_audio_dir)

    # Save updated metadata
    solicited_metadata.to_csv(os.path.join(PROCESSED_DIR, "solicited_metadata_updated.csv"), index=False)

if __name__ == "__main__":
    main()