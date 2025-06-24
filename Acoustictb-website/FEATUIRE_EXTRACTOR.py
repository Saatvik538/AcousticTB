import os
#disable numba caching to avoid permission issues
numba_cache_dir = os.path.join(os.getcwd(), 'numba_cache')
if not os.path.isdir(numba_cache_dir):
    os.makedirs(numba_cache_dir, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = numba_cache_dir

import tempfile
import pandas as pd
import numpy as np
import random
import librosa #for loading audio and mel-spectrogram
import shutil
from tqdm import tqdm

#set a custom temp directory in the current working directory
temp_dir = os.path.join(os.getcwd(), "temp_audio_processing")
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir

# Paths
RAW_DIR = os.path.join("data", "raw")
SOLICITED_META_DIR = os.path.join(RAW_DIR, "solicited_coughs", "metadata")
PROCESSED_DIR = os.path.join("data", "processed")
SOLICITED_META_FILE = os.path.normpath(os.path.join(SOLICITED_META_DIR, "CODA_TB_Solicited_Meta_Info.csv"))
CLINICAL_META_FILE = os.path.normpath(os.path.join(SOLICITED_META_DIR, "CODA_TB_Clinical_Meta_Info.csv"))

#core feature extraction

def add_noise(audio, noise_type, noise_factor=0.005):
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_factor, len(audio))
        return audio + noise
    elif noise_type == "pink":
        pink = np.arange(len(audio), 0, -1).astype(float)
        pink /= pink.sum()
        return audio + pink * np.random.normal(0, noise_factor, len(audio))
    elif noise_type == "ambient":
        ambient = np.sin(2 * np.pi * np.linspace(0, 10, len(audio)))
        return audio + ambient * noise_factor
    return audio


def extract_features(audio_path, sample_rate=16000, add_noise_prob=0.3):
    #load
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    #only keep first 0.5 seconds for cough snippet
    max_samples = int(sample_rate * 0.5)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    
    #noise injection
    if random.random() < add_noise_prob:
        noise_type = random.choice(["gaussian", "pink", "ambient"])
        audio = add_noise(audio, noise_type)
    else:
        noise_type = "none"

    #mel-spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmax=8000)

    #transpose and ensure freq bins
    mel = mel.T
    if mel.shape[1] != 64:
        mel = mel[:, :64]

    return mel, noise_type


def process_metadata(metadata_file, clinical_file, audio_dir, process_all=True):
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata = pd.read_csv(metadata_file)
    clinical_data = pd.read_csv(clinical_file)
    merged = pd.merge(metadata, clinical_data, on='participant', how='left')
    new_rows = []
    out_dir = os.path.join(PROCESSED_DIR, 'features')
    os.makedirs(out_dir, exist_ok=True)

    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc='Processing'):
        fname = row.filename
        path = os.path.join(audio_dir, fname)
        if not os.path.exists(path):
            continue
        feats, noise = extract_features(path)
        base = os.path.splitext(fname)[0]
        ffile = base + (f"_{noise}_noise.npy" if noise!="none" else ".npy")
        dest = os.path.join(out_dir, ffile)
        np.save(dest, feats)
        new = row.copy()
        new['feature_path'] = dest
        new['noise_type'] = noise
        new_rows.append(new)
    return pd.DataFrame(new_rows)

if __name__ == '__main__':
    df = process_metadata(SOLICITED_META_FILE, CLINICAL_META_FILE, os.path.join(RAW_DIR, 'solicited_coughs', 'audio'))
    df.to_csv(os.path.join(PROCESSED_DIR, 'solicited_features.csv'), index=False)
