import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import time
import pygame
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import queue
import xgboost as xgb
import joblib
from scipy import signal

class CoughRecorder:
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 0.5
        self.channels = 1
        self.countdown_duration = 3
        self.prob = 100
        
        pygame.mixer.init()
        
        self.output_dir = "recorded_coughs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model = joblib.load('tb_detection_model.joblib')
        self.scaler = joblib.load('scaler.joblib')
        
        self.root = tk.Tk()
        self.root.title("TB Cough Detector")
        self.setup_gui()
        
        self.audio_queue = queue.Queue()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Clinical Information Section
        clinical_frame = ttk.LabelFrame(main_frame, text="Clinical Information", padding="10")
        clinical_frame.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Age input
        ttk.Label(clinical_frame, text="Age (years):").grid(row=0, column=0, padx=5, pady=2)
        self.age_var = tk.StringVar(value="35")
        self.age_entry = ttk.Entry(clinical_frame, textvariable=self.age_var, width=10)
        self.age_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # Height input
        ttk.Label(clinical_frame, text="Height (cm):").grid(row=1, column=0, padx=5, pady=2)
        self.height_var = tk.StringVar(value="170")
        self.height_entry = ttk.Entry(clinical_frame, textvariable=self.height_var, width=10)
        self.height_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # Weight input
        ttk.Label(clinical_frame, text="Weight (kg):").grid(row=2, column=0, padx=5, pady=2)
        self.weight_var = tk.StringVar(value="70")
        self.weight_entry = ttk.Entry(clinical_frame, textvariable=self.weight_var, width=10)
        self.weight_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # Cough duration input
        ttk.Label(clinical_frame, text="Cough Duration (days):").grid(row=3, column=0, padx=5, pady=2)
        self.cough_duration_var = tk.StringVar(value="7")
        self.cough_duration_entry = ttk.Entry(clinical_frame, textvariable=self.cough_duration_var, width=10)
        self.cough_duration_entry.grid(row=3, column=1, padx=5, pady=2)
        
        # Recording controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, pady=10)
        
        self.record_button = ttk.Button(control_frame, text="Start Recording", command=self.validate_and_start_recording)
        self.record_button.grid(row=0, column=0, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(control_frame, length=200, mode='determinate', variable=self.progress_var)
        self.progress.grid(row=1, column=0, pady=10)
        
        self.status_var = tk.StringVar(value="Ready to record")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.grid(row=2, column=0, pady=10)
        
        self.results_var = tk.StringVar(value="")
        self.results_label = ttk.Label(control_frame, textvariable=self.results_var)
        self.results_label.grid(row=3, column=0, pady=10)

    def validate_clinical_data(self):
        try:
            age = float(self.age_var.get())
            height = float(self.height_var.get())
            weight = float(self.weight_var.get())
            cough_duration = float(self.cough_duration_var.get())
            
            if not (0 < age < 120):
                raise ValueError("Age must be between 0 and 120 years")
            if not (50 < height < 250):
                raise ValueError("Height must be between 50 and 250 cm")
            if not (20 < weight < 300):
                raise ValueError("Weight must be between 20 and 300 kg")
            if not (0 <= cough_duration < 365):
                raise ValueError("Cough duration must be between 0 and 365 days")
                
            return True
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return False

    def validate_and_start_recording(self):
        if self.validate_clinical_data():
            self.start_recording_session()

    def countdown(self):
        for i in range(self.countdown_duration, 0, -1):
            self.status_var.set(f"Get ready to cough in {i}...")
            self.progress_var.set((self.countdown_duration - i) / self.countdown_duration * 100)
            self.root.update()
            time.sleep(1)
        self.status_var.set("COUGH NOW!")
        self.progress_var.set(100)
        self.root.update()

    def record_audio(self):
        self.status_var.set("Recording...")
        self.root.update()
        
        audio_data = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )
        sd.wait()
        
        self.status_var.set("Recording complete")
        self.root.update()
        
        return audio_data

    def process_audio(self, audio_data):
        # Normalize audio
        audio_data = audio_data.flatten()
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # Generate mel spectrogram with VGGish-like parameters
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_mels=64,
            n_fft=2048,
            hop_length=512,
            window='hann',
            center=True,
            pad_mode='reflect',
            power=2.0
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        
        # Normalize the spectrogram
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-5)
        
        # Reshape to match training dimensions
        mel_spec_db = mel_spec_db.T
        
        # Ensure exact shape match
        target_length = 128
        if mel_spec_db.shape[0] < target_length:
            pad_length = target_length - mel_spec_db.shape[0]
            mel_spec_db = np.pad(mel_spec_db, ((0, pad_length), (0, 0)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:target_length, :]
        
        # Calculate sound prediction score
        sound_score = np.max(np.abs(audio_data))  # Use peak amplitude as sound score
        
        # Get clinical features from UI
        clinical_features = np.array([
            float(self.age_var.get()),
            float(self.height_var.get()),
            float(self.weight_var.get()),
            float(self.cough_duration_var.get()),
            80.0,    # heart rate (using average as placeholder)
            37.0,    # temperature (using normal as placeholder)
            sound_score
        ])
        
        # Flatten and combine features
        mel_features = mel_spec_db.flatten()
        all_features = np.concatenate([mel_features, clinical_features])
        
        # Reshape for model input
        features = all_features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        prediction = self.model.predict_proba(features_scaled)[0]
        return prediction[1]

    def start_recording_session(self):
        self.record_button.state(['disabled'])
        for i in range(5):
            self.status_var.set(f"Preparing for cough {i+1} of 5")
            self.root.update()
            self.countdown()
            
            try:
                audio_data = self.record_audio()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.output_dir, f"cough_{timestamp}.wav")
                sf.write(filename, audio_data, self.sample_rate)
                
                self.status_var.set("Processing audio...")
                self.root.update()
                
                tb_probability = self.process_audio(audio_data)
                
                self.results_var.set(f"Cough {i+1}: TB Probability: {tb_probability:.2%}")
                self.prob = tb_probability
                self.root.update()

            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                self.root.update()
            
            time.sleep(1)
        
        self.record_button.state(['!disabled'])
        self.status_var.set(f"Cough {i+1}: TB Probability: {self.prob:.2%}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    try:
        recorder = CoughRecorder()
        recorder.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        input("Press Enter to exit...")