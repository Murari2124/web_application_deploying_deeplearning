import flask
from flask import Flask, render_template, request, redirect, send_from_directory, send_file
import os
import threading
from flask import Response
from tensorflow import keras
import numpy as np
import plotly.express
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

model = keras.models.load_model(r"model.h5")

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
P_F='positive_frames'

@app.route('/')
def index():
    return render_template('index.html')
camera = cv2.VideoCapture(0)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if not os.path.exists(P_F):
    os.makedirs(P_F)
def process_frame(frame):
    # Preprocess the frame for input to the model
    preprocessed_frame = cv2.resize(frame, (1200, 1200))
    preprocessed_frame1 = cv2.resize(frame, (120, 120))
    # Apply the model on the preprocessed frame
    np_img1 = np.array(preprocessed_frame1)
    np_img2 = np.expand_dims(np_img1, axis=0)
    prediction = model.predict(np_img2)
    np_img = np.array(preprocessed_frame)
    if prediction > 0.49999:
        cv2.rectangle(np_img, (100, 100), (1000, 1000), (0, 0, 255), 2)
        """for k in range(0,1200,120):
                for j in range(0, 240, 120):
                     a = np_img[k:k+120,j:j+120]
                     b = np.expand_dims(a, axis=0)
                     if model.predict(b) >= 0.5:
                         # np_img[k:k+120,j:j+120]=np_img[k:k+120,j:j+120]*255
                         #cv2.rectangle(np_img1, (k, j), (120, 120), (0, 0, 255), 2)
                         np_img[k:k+120,j:j+120, 1] = np_img[k:k+120,j:j+120, 1] / 2
                         np_img[k:k+120,j:j+120, 2] = np_img[k:k+120,j:j+120, 2] / 2"""
        data = Image.fromarray(np_img.astype(np.uint8))
        return np_img, prediction

def StartCamera():
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        # Send the frame to the Flask app
        ## read the camera frame
        success, frame = camera.read()
        frame, pred = process_frame(frame)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/video')
def video():
    return Response(StartCamera(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/upload', methods=['POST'])
def upload():
    videos = request.files['video']
    video_path = os.path.join('uploads', videos.filename)
    videos.save(video_path)
    cap = cv2.VideoCapture(video_path)
    from tensorflow import keras
    model = keras.models.load_model(r"model.h5")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'outputs/output3.mp4'
    out = cv2.VideoWriter(output_path, fourcc, 30, (1200, 1200))
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess the frame for input to the model
        preprocessed_frame = cv2.resize(frame, (1200, 1200))
        preprocessed_frame1 = cv2.resize(frame, (120, 120))
        # Apply the model on the preprocessed frame
        np_img1 = np.array(preprocessed_frame1)
        np_img2 = np.expand_dims(np_img1, axis=0)
        prediction = model.predict(np_img2)

        np_img = np.array(preprocessed_frame)
        if prediction > 0.49999:
            cv2.rectangle(np_img, (100, 100), (1000, 1000), (0, 0, 255), 2)
            """for k in range(0,1200,120):
                for j in range(0, 240, 120):
                     a = np_img[k:k+120,j:j+120]
                     b = np.expand_dims(a, axis=0)
                     if model.predict(b) >= 0.5:
                         # np_img[k:k+120,j:j+120]=np_img[k:k+120,j:j+120]*255
                         #cv2.rectangle(np_img1, (k, j), (120, 120), (0, 0, 255), 2)
                         np_img[k:k+120,j:j+120, 1] = np_img[k:k+120,j:j+120, 1] / 2
                         np_img[k:k+120,j:j+120, 2] = np_img[k:k+120,j:j+120, 2] / 2"""
        data = Image.fromarray(np_img.astype(np.uint8))

        out.write(np_img)
    # Call your deep learning model to process the video and generate the output

    # Save the output video
    return redirect('/download')
@app.route('/download')
def download():
    output_path = 'outputs/output3.mp4'
    return send_file(output_path, as_attachment=True)
if __name__ == '__main__':
    app.run( debug=True)
    # Provide the output video for download



