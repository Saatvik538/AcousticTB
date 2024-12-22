import numpy as np
import os
import pandas as pd
import glob
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib

def pad_or_truncate(feature, target_shape=(128, 64)):
    original_shape = feature.shape
    if original_shape[0] < target_shape[0]:
        padding = target_shape[0] - original_shape[0]
        feature = np.pad(feature, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    elif original_shape[0] > target_shape[0]:
        feature = feature[:target_shape[0], :]
    return feature

def load_features_and_labels(features_csv, clinical_csv, noisy_csv, file_pattern):
    features, labels, clinical_data = [], [], []
    noise_types = []

    features_df = pd.read_csv(features_csv)
    clinical_df = pd.read_csv(clinical_csv)
    noisy_df = pd.read_csv(noisy_csv)

    categorical_columns = ['sex', 'tb_prior', 'tb_prior_Pul', 'tb_prior_Extrapul', 
                           'tb_prior_Unknown', 'hemoptysis', 'weight_loss', 
                           'smoke_lweek', 'fever', 'night_sweats']
    clinical_df = pd.get_dummies(clinical_df, columns=categorical_columns)

    sound_scores = dict(zip(features_df['filename'], features_df['sound_prediction_score']))

    files = glob.glob(file_pattern)
    for file_path in tqdm(files, desc="Loading Features"):
        filename = os.path.basename(file_path)
        filename_without_npy = filename.replace('.npy', '')

        if 'noise' in filename:
            # This is a noisy sample
            matching_row = noisy_df[noisy_df['feature_path'].str.contains(filename)]
            if not matching_row.empty:
                feature = np.load(file_path)
                feature = pad_or_truncate(feature)
                features.append(feature)
                labels.append(int(matching_row['tb_status'].values[0]))
                
                clinical_features = matching_row[['age', 'height', 'weight', 'reported_cough_dur', 'heart_rate', 'temperature']].values[0]
                clinical_features = np.append(clinical_features, matching_row['sound_prediction_score'].values[0])
                clinical_data.append(clinical_features)
                
                noise_types.append(matching_row['noise_type'].values[0])
        else:
            # This is an original sample
            matching_row = features_df[features_df['filename'] == filename_without_npy]
            if not matching_row.empty:
                participant = matching_row['participant'].values[0]
                clinical_row = clinical_df[clinical_df['participant'] == participant]
                
                if not clinical_row.empty:
                    feature = np.load(file_path)
                    feature = pad_or_truncate(feature)
                    features.append(feature)
                    labels.append(int(clinical_row['tb_status'].values[0]))
                    
                    clinical_features = clinical_row[['age', 'height', 'weight', 'reported_cough_dur', 'heart_rate', 'temperature']].values[0]
                    sound_score = sound_scores.get(filename_without_npy, 0)
                    clinical_features = np.append(clinical_features, sound_score)
                    
                    clinical_data.append(clinical_features)
                    noise_types.append('none')

    features_array = np.array(features)
    labels_array = np.array(labels)
    clinical_array = np.array(clinical_data)

    print(f"Loaded {len(features_array)} samples with shape {features_array.shape[1:]} and labels.")
    print("Noise type distribution:", Counter(noise_types))
    return features_array, labels_array, clinical_array, noise_types

def train_model(X_train, y_train):
    neg_samples = np.sum(y_train == 0)
    pos_samples = np.sum(y_train == 1)
    scale_pos_weight = neg_samples / pos_samples

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.1,
        n_estimators=100,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_val, y_val, noise_types_val, feature_names=None):
    y_pred = model.predict(X_val)
    y_pred_prob = model.predict_proba(X_val)[:, 1]

    print("\nOverall Classification Report:")
    print(classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No TB', 'TB'], 
                yticklabels=['No TB', 'TB'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Overall Confusion Matrix')
    plt.show()

    accuracy = accuracy_score(y_val, y_pred)
    auc_roc = roc_auc_score(y_val, y_pred_prob)
    
    print(f"\nOverall Validation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"AUC-ROC Score: {auc_roc:.4f}")
    
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    if feature_names is not None:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]

        plt.figure(figsize=(12, 6))
        plt.title("Top 20 Most Important Features")
        plt.bar(range(20), importances[indices])
        plt.xticks(range(20), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Evaluate performance for each noise type
    noise_types = set(noise_types_val)
    for noise_type in noise_types:
        mask = np.array(noise_types_val) == noise_type
        y_val_noise = y_val[mask]
        y_pred_noise = y_pred[mask]
        
        print(f"\nClassification Report for {noise_type} noise:")
        print(classification_report(y_val_noise, y_pred_noise))
        
        cm_noise = confusion_matrix(y_val_noise, y_pred_noise)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_noise, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No TB', 'TB'], 
                    yticklabels=['No TB', 'TB'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {noise_type} noise')
        plt.show()

    return accuracy, auc_roc

def main():
    features_csv = "data/raw/solicited_coughs/metadata/CODA_TB_Solicited_Meta_Info.csv"
    clinical_csv = "data/raw/solicited_coughs/metadata/CODA_TB_Clinical_Meta_Info.csv"
    noisy_csv = "data/processed/solicited_metadata_with_noise.csv"
    feature_files = "data/processed/features/*.npy"

    X_audio, y, X_clinical, noise_types = load_features_and_labels(features_csv, clinical_csv, noisy_csv, feature_files)

    X_audio_flat = X_audio.reshape(X_audio.shape[0], -1)
    X = np.hstack((X_audio_flat, X_clinical))

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_val, y_train, y_val, noise_types_train, noise_types_val = train_test_split(
        X_scaled, y, noise_types,
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\nTraining samples: {X_train_resampled.shape[0]}, Validation samples: {X_val.shape[0]}")
    print(f"TB positive samples in training after SMOTE: {np.sum(y_train_resampled == 1)}")
    print(f"TB positive samples in validation: {np.sum(y_val == 1)}")
    print("Noise type distribution in training:", Counter(noise_types_train))
    print("Noise type distribution in validation:", Counter(noise_types_val))

    model = train_model(X_train_resampled, y_train_resampled)

    feature_names = ([f'audio_{i}' for i in range(X_audio_flat.shape[1])] + 
                    ['age', 'height', 'weight', 'reported_cough_dur', 'heart_rate', 'temperature', 'sound_prediction_score'])

    evaluate_model(model, X_val, y_val, noise_types_val, feature_names)

    # Save the trained model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'tb_detection_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model and scaler saved successfully!")

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
    print(f"\nCross-validation ROC AUC scores: {cv_scores}")
    print(f"Mean CV ROC AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

if __name__ == "__main__":
    main()
