from flask import Flask, request, render_template, url_for, flash
import logging
import sys
import json
import numpy as np
import librosa
import os
from werkzeug import secure_filename



def sigmoid(z):
    s = 1 / (1 + np.exp(-(z)))    
    return s

def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
#     print(w.shape)
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    # Convert probabilities A[0,i] to actual predictions p[0,i]
#     print(A)
    Y_prediction = (A > 0.5).astype(float)
        
    return Y_prediction

def preprocess(music_file):
    to_predict = []
    y, sr = librosa.load(music_file, mono=True, duration=30, offset=15)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # Append tempo
    to_predict.append(tempo)
    # Append beats
    to_predict.append(beats.shape[0])
    # Append chroma_stft
    to_predict.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    # Append rmse
    to_predict.append(np.mean(librosa.feature.rmse(y=y)))
    # Append spectral centroid
    to_predict.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    # Append spectral bandwidth
    to_predict.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    # Append roll-off
    to_predict.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    # Append zero crossing rate
    to_predict.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    # Append mfccs
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for e in mfcc:
        to_predict.append(np.mean(e))

    return to_predict


UPLOAD_FOLDER = '/tmp/'
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    print('here-1')
    return render_template('index.html')


@app.route('/results', methods=["GET", "POST"])
def results():
    print('here-2')
    if request.method == 'GET':
        predict('here2')
        return render_template('index.html')
    else:
        if 'music' not in request.files:
            return redirect('/')
        music = request.files['music']
        filename = secure_filename(music.filename)
        music.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        music_file = UPLOAD_FOLDER + filename
        if music.filename == '':
            return redirect('/')
        print('here')
        features = preprocess(music_file)
        with open('model.json') as f:
            model_dict = json.load(f)
        w = np.array(model_dict['w'])
        b = np.array(model_dict['b'])
        X = np.array(features).reshape(28, 1)
        y = predict(w, b, X)
        if y == 0:
            genre = 'POP'
        else:
            genre = 'Classical'
        print('here-classify')
        print(genre)
        return render_template('results.html', filename=filename, genre=genre)
        

if __name__ == "__main__":
    app.run(debug=True)