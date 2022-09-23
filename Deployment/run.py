from fastapi import FastAPI
import numpy as np
from model import get_model
import cv2 
import tensorflow as tf

app = FastAPI()

@app.post("/predict")
def get_pred():
    print("***"*15)
    model = get_model()
    model.load_weights('/home/rabi/Desktop/Project2/model.h5')
    img = cv2.imread(r"/home/rabi/Desktop/Project/pic/zero.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)
    newing = tf.keras.utils.normalize (resized, axis = 1) ## 0 to 1 scaling
    newing = np.array(newing).reshape(-1, 28, 28,1) ## kernal operation of convolutional layer
    predictions = model.predict(newing)
    print (np.argmax(predictions))
    print("--------"*15)
   
    return {'prediction':np.argmax(predictions).tolist()}
   # print (np.argmax(predictions))
