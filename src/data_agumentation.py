import os
import shutil
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp_audio_processing"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ["NUMBA_CACHE_DIR"] = str(TEMP_DIR / "numba_cache")
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TMPDIR"] = str(TEMP_DIR)

RAW_AUDIO_DIR = BASE_DIR / "data" / "raw" / "solicited_coughs" / "audio"
META_DIR = BASE_DIR / "data" / "raw" / "solicited_coughs" / "metadata"
OUT_DIR = BASE_DIR / "data" / "processed"
FEAT_DIR = OUT_DIR / "features"

def load_esc50_noise_samples():
    # load environmental sounds from esc-50 dataset
    noise_samples = []
    
    esc50_path = BASE_DIR / "ESC-50"
    csv_path = esc50_path / "meta" / "esc50.csv"
    audio_path = esc50_path / "audio"
    
    if not csv_path.exists() or not audio_path.exists():
        print("ESC-50 dataset not found in project root")
        print("Expected structure: ESC-50/meta/esc50.csv and ESC-50/audio/*.wav")
        return []
    
    # read metadata to get environmental sound classes
    meta_df = pd.read_csv(csv_path)
    
    # environmental sound categories (avoiding human sounds and dog barks)
    env_categories = [
        'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',
        'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
        'footsteps', 'clock_tick', 'keyboard_typing', 'mouse_click', 'door_wood_knock',
        'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm'
    ]
    
    # filter for environmental sounds only
    env_sounds = meta_df[meta_df['category'].isin(env_categories)]
    
    for _, row in env_sounds.iterrows():
        audio_file = audio_path / row['filename']
        if audio_file.exists():
            try:
                # load full 5-second clip at 22050 hz, force mono
                noise, _ = librosa.load(str(audio_file), sr=22050, duration=5.0, mono=True)
                if len(noise) > 0:
                    noise_samples.append(noise)
            except Exception as e:
                print(f"Failed to load {audio_file}: {e}")
                continue
    
    print(f"Loaded {len(noise_samples)} ESC-50 environmental sounds")
    return noise_samples

def add_noise(audio, noise_type='gaussian', intensity=0.01, esc50_samples=None, snr_db=15):
    # add noise at specified SNR level
    if noise_type == 'gaussian':
        noise = np.random.normal(0, intensity, len(audio))
        return audio + noise
    
    elif noise_type == 'environmental' and esc50_samples:
        # use real environmental sounds from esc-50
        if len(esc50_samples) == 0:
            return audio
        
        # randomly select environmental sample (avoid np.random.choice for 2D arrays)
        sample_idx = np.random.randint(0, len(esc50_samples))
        env_sample = esc50_samples[sample_idx]
        
        # ensure env_sample is 1D
        if env_sample.ndim > 1:
            env_sample = env_sample.flatten()
        
        # handle length mismatch - esc50 clips are 5 seconds, coughs are ~0.5 seconds
        if len(env_sample) >= len(audio):
            # randomly select segment from longer environmental sound
            start_idx = np.random.randint(0, len(env_sample) - len(audio) + 1)
            noise = env_sample[start_idx:start_idx + len(audio)]
        else:
            # repeat if environmental sound is shorter than cough
            repeats = int(np.ceil(len(audio) / len(env_sample)))
            noise = np.tile(env_sample, repeats)[:len(audio)]
        
        # scale noise to achieve target snr
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            noise_scale = np.sqrt(signal_power / (noise_power * snr_linear))
            noise = noise * noise_scale
        
        return audio + noise
    
    return audio

def extract_mel_spectrogram(audio, sr=22050, n_mels=64, hop_length=512):
    # extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # resize to 64x64
    if mel_spec_db.shape[1] < 64:
        # pad if too short
        pad_width = 64 - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spec_db.shape[1] > 64:
        # truncate if too long
        mel_spec_db = mel_spec_db[:, :64]
    
    return mel_spec_db

def process_audio_file(audio_path, metadata_row, esc50_samples=None):
    try:
        audio, sr = librosa.load(audio_path, sr=22050)
        
        base_id = Path(audio_path).stem
        
        results = []
        
        # original clean version
        mel_spec = extract_mel_spectrogram(audio, sr)
        clean_path = FEAT_DIR / f"{base_id}_clean.npy"
        np.save(clean_path, mel_spec)
        
        result_row = metadata_row.copy()
        result_row['feature_file'] = f"{base_id}_clean.npy"
        result_row['noise_type'] = 'clean'
        result_row['base_patient_id'] = base_id  
        results.append(result_row)
        
        # augmented versions - esc50 environmental + gaussian
        noise_types = []
        
        if esc50_samples:
            noise_types.append('environmental')
        
        # always include gaussian as backup
        noise_types.append('gaussian')
        
        for noise_type in noise_types:
            noisy_audio = add_noise(audio, noise_type, intensity=0.01, 
                                  esc50_samples=esc50_samples, snr_db=20)
            mel_spec_noisy = extract_mel_spectrogram(noisy_audio, sr)
            noisy_path = FEAT_DIR / f"{base_id}_{noise_type}.npy"
            np.save(noisy_path, mel_spec_noisy)
            
            result_row = metadata_row.copy()
            result_row['feature_file'] = f"{base_id}_{noise_type}.npy"
            result_row['noise_type'] = noise_type
            result_row['base_patient_id'] = base_id  
            results.append(result_row)
        
        return results
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return []

def process_all():
    # load metadata
    metadata_path = META_DIR / "CODA_TB_Solicited_Meta_Info.csv"
    metadata = pd.read_csv(metadata_path)
    
    # load esc-50 environmental sounds
    print("Loading ESC-50 environmental sounds...")
    esc50_samples = load_esc50_noise_samples()
    if esc50_samples:
        print(f"Loaded {len(esc50_samples)} ESC-50 environmental sounds")
    else:
        print("No ESC-50 dataset found. Using Gaussian noise only.")
    
    all_results = []
    
    # process each audio file
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing audio files"):
        audio_filename = row['filename']
        audio_path = RAW_AUDIO_DIR / audio_filename
        
        if audio_path.exists():
            results = process_audio_file(audio_path, row, esc50_samples)
            all_results.extend(results)
        else:
            print(f"Audio file not found: {audio_path}")
    
    return pd.DataFrame(all_results)

def main():
    numba_cache = TEMP_DIR / "numba_cache"
    numba_cache.mkdir(exist_ok=True)
    
    if FEAT_DIR.exists():
        # windows-compatible directory removal
        for file in FEAT_DIR.rglob('*'):
            if file.is_file():
                file.unlink()
        try:
            shutil.rmtree(FEAT_DIR)
        except Exception:
            pass
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    out_df = process_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_DIR / "solicited_metadata_with_noise.csv", index=False)

    print(f"Created {len(out_df)} feature files")
    print(f"Unique patients: {out_df['base_patient_id'].nunique()}")
    print("File count breakdown:")
    print(out_df['noise_type'].value_counts())
    
    shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()
