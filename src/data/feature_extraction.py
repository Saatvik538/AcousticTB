import tensorflow_hub as hub
import tensorflow as tf
import librosa
import numpy as np
import os

# Constants
SAMPLE_RATE = 16000
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"

# Load VGGish model
print("Loading VGGish model from TensorFlow Hub...")
vggish = hub.load(VGGISH_MODEL_URL)

def load_audio(file_path, sample_rate=SAMPLE_RATE):
    """Load an audio file and resample to the target sample rate."""
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return audio

def extract_vggish_features(audio):
    """Extract VGGish embeddings from audio."""
    # Ensure the audio is a 2D tensor with shape [batch_size, num_samples]
    audio = np.expand_dims(audio, axis=0)
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    # Extract features
    embeddings = vggish(audio_tensor)
    return embeddings.numpy()

def process_audio_files(audio_dir, output_dir):
    """
    Process all .wav files in a directory and extract features.

    Args:
        audio_dir (str): Directory containing .wav files.
        output_dir (str): Directory to save extracted features.
    """
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                try:
                    audio = load_audio(file_path)
                    features = extract_vggish_features(audio)

                    # Save features
                    feature_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_features.npy")
                    np.save(feature_file, features)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")