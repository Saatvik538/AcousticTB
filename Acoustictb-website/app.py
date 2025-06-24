from flask import Flask, render_template, request, jsonify
import os, subprocess, numpy as np, joblib, shutil
import soundfile as sf  #already installed
import threading, subprocess
from FEATUIRE_EXTRACTOR import extract_features
from FINAL_MODEL import pad_or_truncate

FFMPEG_BIN = r"C:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
if not os.path.isfile(FFMPEG_BIN):
    raise RuntimeError(f"ffmpeg not found at {FFMPEG_BIN}")

app = Flask(__name__)
UPLOAD = app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(UPLOAD, exist_ok=True)

model  = joblib.load('tb_detection_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #save upload
    f = request.files.get('coughfile')
    if not f or f.filename == '':
        return jsonify(error="No file uploaded"), 400
    raw = os.path.join(UPLOAD, f.filename)
    f.save(raw)

    #convert if needed
    ext = os.path.splitext(raw)[1].lower()
    if ext in ('.webm','.ogg','.m4a','.mp4'):
        wav = raw.rsplit('.',1)[0] + '.wav'
        cmd = [FFMPEG_BIN,'-y','-i', raw,'-ar','16000','-ac','1', wav]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        except Exception as e:
            return jsonify(error="FFmpeg failed", details=str(e)), 500
        audio = wav
    else:
        audio = raw

    #extract features
    try:
        feats, noise = extract_features(audio)
    except Exception as e:
        return jsonify(error="Feature extraction failed", details=str(e)), 500

    #pad, flatten, append dummy clinical, scale and predict
    feats = pad_or_truncate(feats).flatten().reshape(1,-1)
    clinical = np.array([[25,170,65,21,80,36.5,0.9]])
    X = np.hstack((feats, clinical))
    prob = model.predict_proba(scaler.transform(X))[0][1]
    label = "TB Positive" if prob>0.5 else "TB Negative"

    return jsonify(result=label, confidence=f"{prob*100:.1f}%", noise_type=noise)
def start_flower_client():
    subprocess.run(["python", "fl_client.py"])
if __name__ == '__main__':
    subprocess.run(["python","fl_simulation.py"])
    app.run(debug=True)
