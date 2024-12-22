import numpy as np
import os
import xgboost as xgb
import pandas as pd
import glob
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Function to pad or truncate the features to a fixed size
def pad_or_truncate(feature, target_shape=(128, 64)):
    original_shape = feature.shape
    if original_shape[0] < target_shape[0]:
        padding = target_shape[0] - original_shape[0]
        feature = np.pad(feature, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    elif original_shape[0] > target_shape[0]:
        feature = feature[:target_shape[0], :]
    return feature

# Function to load and preprocess features with clinical labels
def load_features_and_labels(features_csv, clinical_csv, file_pattern):
    features, labels = [], []

    # Load metadata files
    features_df = pd.read_csv(features_csv)
    clinical_df = pd.read_csv(clinical_csv)

    # Create a dictionary for quick lookup of tb_status
    tb_status_dict = dict(zip(clinical_df['participant'], clinical_df['tb_status']))

    # Match .npy feature files with their corresponding labels
    files = glob.glob(file_pattern)
    for file_path in tqdm(files, desc="Loading Features"):
        filename = os.path.basename(file_path)
        filename_without_npy = filename.replace('.npy', '')  # Remove .npy extension

        # Find the matching row in features_df
        matching_row = features_df[features_df['filename'] == filename_without_npy]

        if not matching_row.empty:
            participant = matching_row['participant'].values[0]
            
            # Get TB status from the dictionary
            if participant in tb_status_dict:
                tb_status = int(tb_status_dict[participant])

                # Load the .npy feature file
                feature = np.load(file_path)
                feature = pad_or_truncate(feature)

                features.append(feature)
                labels.append(tb_status)
            else:
                print(f"No clinical metadata found for participant {participant}, skipping.")
        else:
            print(f"No metadata found for {filename}, skipping.")
    
    features_array = np.array(features)
    labels_array = np.array(labels)

    print(f"Loaded {len(features_array)} samples with shape {features_array.shape[1:]} and labels.")
    return features_array, labels_array

# Train XGBoost model
def train_model(X_train, y_train, X_val, y_val):
    X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten for XGBoost
    X_val = X_val.reshape(X_val.shape[0], -1)

    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

# Evaluate model and visualize
def evaluate_model(model, X_val, y_val):
    X_val = X_val.reshape(X_val.shape[0], -1)
    y_pred = model.predict(X_val)
    y_pred_prob = model.predict_proba(X_val)[:, 1]  # Probabilities for ROC (optional)

    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No TB', 'TB'], yticklabels=['No TB', 'TB'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    return accuracy

# Plot feature importance
def plot_feature_importance(model):
    plt.figure(figsize=(8, 6))
    xgb.plot_importance(model, importance_type='weight', max_num_features=10)
    plt.title('Top 10 Feature Importances')
    plt.show()

# Main function
def main():
    # Define file paths
    features_csv = "data/raw/solicited_coughs/metadata/CODA_TB_Solicited_Meta_Info.csv"
    clinical_csv = "data/raw/solicited_coughs/metadata/CODA_TB_Clinical_Meta_Info.csv"
    feature_files = "data/processed/features/*.npy"

    # Load features and labels
    X, y = load_features_and_labels(features_csv, clinical_csv, feature_files)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Train model
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate model
    accuracy = evaluate_model(model, X_val, y_val)

    # Feature importance
    plot_feature_importance(model)

    print(f"Final Model Validation Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
