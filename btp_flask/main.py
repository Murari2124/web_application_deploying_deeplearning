import numpy as np
import plotly.express
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

import cv2




model = keras.models.load_model(r"model.h5")
P_F='positive_frames'
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
if __name__ == "__main__":
    frame_count = 0
    n = input("Print the value of n:")
    video_path = r"C:\Users\murar\Desktop\YouCut_20230430_151205595.mp4"
    camera = cv2.VideoCapture(video_path)


    #camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        modified_frame, predictions = process_frame(frame)
        # Save positive predictions to a folder
        output_filename = "frame_{}.jpg".format(frame_count)
        frame_count+=1
        if predictions > 0.499999:
            cv2.imwrite(os.path.join(P_F, output_filename), modified_frame)
            cv2.imshow("Frame", modified_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            # Release the capture and destroy all windows
    camera.release()



    def rec(img,l,b,i,j):
        if(model.predict(img)<0.5):
            return img
        if(l==120 and b==120):
            if(model.predict(img)>=0.5):
                img[:,:,1]=img[:,:,1]/2
                return img
        else:
            img[i:l/2,j:b/2]=rec(img[i:l/2,j:b/2],l/2,b/2,i,j)
            img[l/2:l,j:b/2]=rec(img[l/2:l,j:b/2],l,b/2,l/2,j)
            img[i:l/2,b/2:b]=rec(img[i:l/2,b/2:b],l/2,b,i,b/2)
            img[l/2:l,b/2:b]=rec(img[l/2:l,b/2:b],l,b,l/2,b/2)
        return img










    img=rec(img,1200,1200,0,0)
