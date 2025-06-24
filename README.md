# AcousticTB
AcousticTB: A machine learning framework leveraging XGBoost and audio signal processing for detecting tuberculosis from cough recordings.


**AcousticTB** is a hybrid machine learning framework designed for the augmented detection of tuberculosis using cough audio recordings and clinical data. This project utilizes the XGBoost model and integrates audio signal processing, noise handling, and clinical features for enhanced prediction accuracy.

## Features
- Records and analyzes cough sounds using the `cough_recorder` module.
- Leverages an XGBoost-based binary classifier for tuberculosis detection.
- Handles noisy data samples and integrates clinical metadata for predictions.
- Provides detailed evaluation metrics such as confusion matrices and ROC curves.
- Offers a prototype **web application** for real-time audio capture and diagnosis.
- Includes a federated learning demo using **Flower** for privacy-aware distributed training.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Saatvik538/AcousticTB.git
   cd AcousticTB
2. Website:
   cd Acoustictb-website
   python app.py
   Open: http://127.0.0.1:5000
3. FL Demo Files:
   fl_server.py: Starts the Flower server to coordinate training.
   fl_client.py: Client that trains locally on pre-extracted features.
   fl_simulation.py: Optional simulation runner using Flower’s VCE engine.
   The final model is saved as global_after_fl_demo.joblib
