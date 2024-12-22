import numpy as np
import os
import pandas as pd
import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def pad_or_truncate(feature, target_shape=(128, 64)):
    original_shape = feature.shape
    if original_shape[0] < target_shape[0]:
        padding = target_shape[0] - original_shape[0]
        feature = np.pad(feature, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    elif original_shape[0] > target_shape[0]:
        feature = feature[:target_shape[0], :]
    return feature

def load_features_and_labels(features_csv, clinical_csv, file_pattern):
    features, labels, clinical_data = [], [], []

    features_df = pd.read_csv(features_csv)
    clinical_df = pd.read_csv(clinical_csv)

    # Convert categorical variables to numerical
    clinical_df = pd.get_dummies(clinical_df, columns=['sex', 'tb_prior', 'tb_prior_Pul', 'tb_prior_Extrapul', 
                                                       'tb_prior_Unknown', 'hemoptysis', 'weight_loss', 
                                                       'smoke_lweek', 'fever', 'night_sweats'])

    files = glob.glob(file_pattern)
    for file_path in tqdm(files, desc="Loading Features"):
        filename = os.path.basename(file_path)
        filename_without_npy = filename.replace('.npy', '')

        matching_row = features_df[features_df['filename'] == filename_without_npy]

        if not matching_row.empty:
            participant = matching_row['participant'].values[0]
            clinical_row = clinical_df[clinical_df['participant'] == participant]
            
            if not clinical_row.empty:
                feature = np.load(file_path)
                feature = pad_or_truncate(feature)

                features.append(feature)
                labels.append(int(clinical_row['tb_status'].values[0]))
                
                clinical_features = clinical_row.drop(['participant', 'tb_status'], axis=1).values[0]
                clinical_data.append(clinical_features)
            else:
                print(f"No clinical data found for participant {participant}, skipping.")
        else:
            print(f"No metadata found for {filename}, skipping.")
    
    features_array = np.array(features)
    labels_array = np.array(labels)
    clinical_array = np.array(clinical_data)

    print(f"Loaded {len(features_array)} samples with shape {features_array.shape[1:]} and labels.")
    return features_array, labels_array, clinical_array

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_pred_prob = model.predict_proba(X_val)[:, 1]

    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No TB', 'TB'], yticklabels=['No TB', 'TB'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    accuracy = accuracy_score(y_val, y_pred)
    auc_roc = roc_auc_score(y_val, y_pred_prob)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"AUC-ROC Score: {auc_roc:.4f}")

    return accuracy, auc_roc

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def main():
    features_csv = "data/raw/solicited_coughs/metadata/CODA_TB_Solicited_Meta_Info.csv"
    clinical_csv = "data/raw/solicited_coughs/metadata/CODA_TB_Clinical_Meta_Info.csv"
    feature_files = "data/processed/features/*.npy"

    X_audio, y, X_clinical = load_features_and_labels(features_csv, clinical_csv, feature_files)

    X_audio_flat = X_audio.reshape(X_audio.shape[0], -1)
    X = np.hstack((X_audio_flat, X_clinical))

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Training samples: {X_train_resampled.shape[0]}, Validation samples: {X_val.shape[0]}")

    model = train_model(X_train_resampled, y_train_resampled)

    accuracy, auc_roc = evaluate_model(model, X_val, y_val)

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

    feature_names = [f'audio_{i}' for i in range(X_audio_flat.shape[1])] + list(pd.get_dummies(pd.read_csv(clinical_csv)).columns[1:-1])
    plot_feature_importance(model, feature_names)

    print(f"Final Model Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Model AUC-ROC Score: {auc_roc:.4f}")

if __name__ == "__main__":
    main()