import numpy as np
import os
import pandas as pd
import glob
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

    categorical_columns = ['sex', 'tb_prior', 'tb_prior_Pul', 'tb_prior_Extrapul', 
                           'tb_prior_Unknown', 'hemoptysis', 'weight_loss', 
                           'smoke_lweek', 'fever', 'night_sweats']
    clinical_df = pd.get_dummies(clinical_df, columns=categorical_columns)

    sound_scores = dict(zip(features_df['filename'], features_df['sound_prediction_score']))

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
                sound_score = sound_scores.get(filename_without_npy, 0)
                clinical_features = np.append(clinical_features, sound_score)
                
                clinical_data.append(clinical_features)

    features_array = np.array(features)
    labels_array = np.array(labels)
    clinical_array = np.array(clinical_data)

    print(f"Loaded {len(features_array)} samples with shape {features_array.shape[1:]} and labels.")
    return features_array, labels_array, clinical_array

def create_model(input_shape, n_clinical_features):
    model = Sequential([
        # CNN layers for audio features
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Merge with clinical features
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model

def train_model(X_audio_train, X_clinical_train, y_train, X_audio_val, X_clinical_val, y_val):
    input_shape = X_audio_train.shape[1:]
    n_clinical_features = X_clinical_train.shape[1]
    
    model = create_model(input_shape, n_clinical_features)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    
    history = model.fit(
        [X_audio_train, X_clinical_train], y_train,
        validation_data=([X_audio_val, X_clinical_val], y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight={0: 1, 1: len(y_train) / (2 * np.sum(y_train))}
    )
    
    return model, history

def evaluate_model(model, X_audio_val, X_clinical_val, y_val):
    y_pred_prob = model.predict([X_audio_val, X_clinical_val])
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nClassification Report:")
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
    
    print(f"\nValidation Metrics:")
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

    return accuracy, auc_roc

def main():
    features_csv = "data/raw/solicited_coughs/metadata/CODA_TB_Solicited_Meta_Info.csv"
    clinical_csv = "data/raw/solicited_coughs/metadata/CODA_TB_Clinical_Meta_Info.csv"
    feature_files = "data/processed/features/*.npy"

    X_audio, y, X_clinical = load_features_and_labels(features_csv, clinical_csv, feature_files)

    # Reshape audio features for CNN input
    X_audio = X_audio.reshape(X_audio.shape[0], X_audio.shape[1], X_audio.shape[2], 1)

    # Handle missing values in clinical data
    imputer = SimpleImputer(strategy='mean')
    X_clinical = imputer.fit_transform(X_clinical)

    # Scale clinical features
    scaler = StandardScaler()
    X_clinical = scaler.fit_transform(X_clinical)

    # Split data
    X_audio_train, X_audio_val, X_clinical_train, X_clinical_val, y_train, y_val = train_test_split(
        X_audio, X_clinical, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    print(f"\nTraining samples: {X_audio_train.shape[0]}, Validation samples: {X_audio_val.shape[0]}")
    print(f"TB positive samples in training: {np.sum(y_train == 1)}")
    print(f"TB positive samples in validation: {np.sum(y_val == 1)}")

    # Train model
    model, history = train_model(X_audio_train, X_clinical_train, y_train, X_audio_val, X_clinical_val, y_val)

    # Evaluate model
    accuracy, auc_roc = evaluate_model(model, X_audio_val, X_clinical_val, y_val)

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()