import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

def p_or_t(feature: np.ndarray, target_shape=(128, 64)) -> np.ndarray:
    # pad or truncate mel spectrogram to target shape
    h, w = feature.shape
    th, tw = target_shape
    if h < th:
        feature = np.pad(feature, ((0, th - h), (0, 0)), mode='constant')
    else:
        feature = feature[:th, :]
    return feature

def composite(y_true, y_pred):
    # composite score for tb detection: 0.56*sensitivity + 0.44*specificity
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return 0.56 * sensitivity + 0.44 * specificity

def load_all_from_processed(noisy_csv: str):
    # load processed features with noise augmentation and base patient IDs
    df = pd.read_csv(noisy_csv)
    print(f"Loaded augmented CSV with columns: {df.columns.tolist()}")
    
    # load original metadata to get TB status and clinical data
    meta_path = Path(noisy_csv).parent.parent / "raw" / "solicited_coughs" / "metadata" / "CODA_TB_Solicited_Meta_Info.csv"
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        print(f"Loaded original metadata with columns: {meta_df.columns.tolist()}")
    else:
        print(f"Original metadata not found at {meta_path}")
        meta_df = None
    
    # initialize data containers
    spectrograms = []
    labels = []
    clinical_data = []
    noise_types = []
    base_patient_ids = []
    
    print(f"Processing {len(df)} entries...")
    
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"Processed {i}/{len(df)} entries...")
            
        # construct path to feature file
        feat_path = Path(noisy_csv).parent / "features" / row['feature_file']
        if feat_path.exists():
            spec = np.load(feat_path)
            spectrograms.append(spec)
            
            # get TB status from original metadata if available
            tb_status = 0
            if meta_df is not None:
                # match by participant ID
                participant_matches = meta_df[meta_df['participant'] == row['participant']]
                if not participant_matches.empty:
                    participant_row = participant_matches.iloc[0]
                    # check for TB status in various possible columns
                    if 'cough_detected' in participant_row and pd.notna(participant_row['cough_detected']):
                        tb_status = int(participant_row['cough_detected'])
                    elif 'tb_status' in participant_row and pd.notna(participant_row['tb_status']):
                        tb_status = int(participant_row['tb_status'])
                    elif 'diagnosis' in participant_row and pd.notna(participant_row['diagnosis']):
                        tb_status = 1 if 'TB' in str(participant_row['diagnosis']).upper() else 0
                    elif 'TB' in participant_row and pd.notna(participant_row['TB']):
                        tb_status = int(participant_row['TB'])
            
            labels.append(tb_status)
            noise_types.append(row['noise_type'])
            base_patient_ids.append(row['base_patient_id'])
            
            # extract clinical features from original metadata
            clinical_features = [0.0] * 7  # default values
            if meta_df is not None:
                participant_matches = meta_df[meta_df['participant'] == row['participant']]
                if not participant_matches.empty:
                    participant_row = participant_matches.iloc[0]
                    clinical_features = [
                        float(participant_row.get('age', 0)) if pd.notna(participant_row.get('age', 0)) else 0.0,
                        float(participant_row.get('height', 0)) if pd.notna(participant_row.get('height', 0)) else 0.0,
                        float(participant_row.get('weight', 0)) if pd.notna(participant_row.get('weight', 0)) else 0.0,
                        float(participant_row.get('reported_cough_dur', 0)) if pd.notna(participant_row.get('reported_cough_dur', 0)) else 0.0,
                        float(participant_row.get('heart_rate', 0)) if pd.notna(participant_row.get('heart_rate', 0)) else 0.0,
                        float(participant_row.get('temperature', 0)) if pd.notna(participant_row.get('temperature', 0)) else 0.0,
                        float(row.get('sound_prediction_score', 0)) if pd.notna(row.get('sound_prediction_score', 0)) else 0.0
                    ]
            clinical_data.append(clinical_features)
        else:
            print(f"Feature file not found: {feat_path}")
    
    X_spec = np.array(spectrograms)
    y = np.array(labels)
    X_clin = np.array(clinical_data)
    
    print(f"Final data shapes:")
    print(f"  Spectrograms: {X_spec.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Clinical: {X_clin.shape}")
    print(f"  Noise distribution: {Counter(noise_types)}")
    print(f"  Unique base patients: {len(set(base_patient_ids))}")
    print(f"  TB status distribution: {Counter(y)}")
    
    return X_spec, y, X_clin, noise_types, base_patient_ids