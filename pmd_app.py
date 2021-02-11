import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(96, 96),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
    
def skin_detector(image_data):
    size = (150,150) 
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # skin color values
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)
    
    imageYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    # filter pixels in skin color range
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)
    # count non black pixels
    nb_cnt = cv2.countNonZero(cv2.cvtColor(skinYCrCb, cv2.COLOR_BGR2GRAY ) )
    # get pixel count of image
    height, width, channels = skinYCrCb.shape
    cntPixels = height*width
    ratio_skin = 100*(nb_cnt/cntPixels)
    return ratio_skin


model = tf.keras.models.load_model('MobileNetV2_model.h5')

st.write("""
         # Photo Manipulation Prediction
         """
         )

st.write("This is a simple image classification web app to predict photo manipulation")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    skin_ratio = skin_detector(image)
    
    if np.argmax(prediction) == 0:
        st.write("It could be real!")
    if np.argmax(prediction) == 1:
        st.write("It could be manipulated!")
    if skin_ratio > 60:
        st.write('Skin detected')
    if skin_ratio <= 60:
        st.write('Not enough human skin tone surface')
    
    st.text("Probability (0: Real, 1: Fake)")
    st.write(prediction)
