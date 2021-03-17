#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st

#import tensorflow as tf
from PIL import Image
import requests
import base64
from io import BytesIO

main_bg = "background.jpg"
main_bg_ext = "jpg"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob
import imutils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import numpy as np


# In[2]:



def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)
        
wpod_net_path = "model/wpod-net.json"
wpod_net = load_model(wpod_net_path)
def preprocess_image(image_path,resize=False):
    img = np.array(image_path.convert('RGB'))
    #img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction
# Load model architecture, weight and labels
json_file = open('model/MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model/License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('model/license_character_classes.npy')
#img_y1=img_to_array(img)
#img_y1=np.expand_dims(img_y1, axis=0)
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

st.sidebar.info("This is an Licence plate detection  web deployment Model.")

st.set_option('deprecation.showfileUploaderEncoding', False)

#st.title("Image Captioning")
st.markdown("<h1 style='text-align: center; color: green;'>Licence Plate Detection Model</h1>", unsafe_allow_html=True)
st.write("")

st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
status = st.radio("Hello, Do you want to Upload an Image or Insert an Image URL?",("Upload Image","Insert URL"))
if status == 'Upload Image':
    st.success("Please Upload an Image")
    file_up = st.file_uploader("Upload an image", type="jpg")
    if file_up is not None:
            image = Image.open(file_up)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Just a second...")
            vehicle, LpImg,cor = get_plate(image)
            if (len(LpImg)): #check if there is at least one license image
                # Scales, calculates absolute values, and converts the result to 8-bit.
                plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    
                # convert to grayscale and blur the image
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray,(7,7),0)
                
                # Applied inversed thresh_binary 
                binary = cv2.threshold(blur, 180, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                
                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            keypoints = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            #contours = sorted(contours, key= cv2.contourArea, reverse=True)
            contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0], reverse = False)
            test_roi=plate_image.copy()
            
            st.image(LpImg[0], caption = "Licence Plate Detected", use_column_width =False)
            col1, col2 = st.beta_columns(2)

            col1.header("blur")
            col1.image(blur, use_column_width=True)
            
            col2.header("Grayscale")            
            col2.image(gray, use_column_width=True)
            col1, col2 = st.beta_columns(2)

            col1.header("binary")
            col1.image(binary, use_column_width=True)
            
            col2.header("dilation")            
            col2.image(thre_mor, use_column_width=True)
            crop_characters=[]
            for c in contours:
                digit_w,digit_h=30,60
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w
                if 1<=ratio<=3.5: # Only select contour with defined ratio
                    if h/plate_image.shape[0]>0.5: # Select contour which has the height larger than 50% of the plate
                        # Draw bounding box arroung digit number
                        cv2.rectangle(test_roi, (x+1, y+1), ((x+1) + (w+1), (y+1) + (h+1)), (0, 255,0), 2)
            
                        # Sperate number and gibe prediction
                        curr_num = thre_mor[y:y+h,x:x+w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)
            st.write("Detect {} letters...".format(len(crop_characters)))
            st.image(test_roi)
            #fig = plt.figure(figsize=(15,3))
            #cols = len(crop_characters)
            #grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

            final_string = ''
            for i,character in enumerate(crop_characters):
                #fig.add_subplot(grid[i])
                title = np.array2string(predict_from_model(character,model,labels))
                #plt.title('{}'.format(title.strip("'[]"),fontsize=20))
                final_string+=title.strip("'[]")
                #plt.axis(False)
                #plt.imshow(character,cmap='gray')

            st.write(final_string)
            

  
else:
    st.success("Please Insert Web URL")
    url = st.text_input("Insert URL below")
    if url:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Just a second...")
        vehicle, LpImg,cor = get_plate(image)
        st.image(LpImg[0], caption = "Licence Plate Detected", use_column_width =True)




