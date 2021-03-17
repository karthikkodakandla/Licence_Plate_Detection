# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:21:07 2020

@author: Vandhana
"""

import streamlit as st

#import tensorflow as tf
#from PIL import Image
import requests
import base64
from io import BytesIO

main_bg = "background.jpg"
main_bg_ext = "jpg"

test_image_path = "test1.jpg"
lpdetect.vehicle, LpImg,cor = lpdeget_plate(test_image_path)




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

st.sidebar.info("This is an Image Captioning web deployment Model.The application identifies the objects in \
                the picture and generates Caption. It was built using a Convolution Neural Network (CNN) for object identification and RNN to generate captions using sequence to sequence model (LSTM)")

st.set_option('deprecation.showfileUploaderEncoding', False)

#st.title("Image Captioning")
st.markdown("<h1 style='text-align: center; color: green;'>Image Captioning</h1>", unsafe_allow_html=True)
st.write("")

st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
status = st.radio("Hello, Do you want to Upload an Image or Insert an Image URL?",("Upload Image","Insert URL"))
if status == 'Upload Image':
    st.success("Please Upload an Image")
    file_up = st.file_uploader("Upload an image", type="jpg")
    if file_up is not None:
            #image = Image.open(file_up)
            #st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Just a second...")
            vehicle, LpImg,cor = lpdetect.get_plate(file_up)
            st.image(LpImg[0], caption = "Licence Plate Detected", use_column_width =True)

  
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
        
