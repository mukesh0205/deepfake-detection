import tensorflow as tf
import dlib
import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import streamlit as st 
from detect import detect
import pyttsx3

def main(model):
    st.set_page_config(page_title="Deep Fake Detection", page_icon=":guardsman:", layout="wide")
    # Set the background image
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://www.google.com/search?q=deep+fake+4k+&tbm=isch#imgrc=sTMDg8Sg9L6DrM");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    st.title('Deep Fake Detection')
    up_vdo = st.file_uploader('Upload a Video', type=['mp4'])
    if up_vdo is not None:
        with open('uploaded_video.mp4', 'wb') as f:
            f.write(up_vdo.getvalue())
            
        result = detect(model, 'uploaded_video.mp4')
        if result==0:
            st.warning('Original Video')
            engine = pyttsx3.init()
            engine.say("The uploaded video is an original video")
            engine.runAndWait()
        else:
            st.warning('Fake Video')
            engine = pyttsx3.init()
            engine.say("The uploaded video is a fake video")
            engine.runAndWait()
        
        # Add a preview of the uploaded video
        video_file = open('uploaded_video.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

if __name__=='__main__':
    model = load_model('deepfake-detection-model.h5')
    main(model)
