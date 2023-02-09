import streamlit as st
import cv2
import numpy as np
from skimage import io, restoration
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
#import plotly.express as px
import pickle
from skimage.io import imread
import os
from PIL import Image

model_final = load_model('model.h5')

Races=pickle.load(open('Races','rb'))

def process_image(image):
    if image is None:
        st.write("L'image est vide.")
        return None

    Image224 = cv2.resize(np.array(image), (224,224))
    gray = cv2.cvtColor(Image224, cv2.COLOR_BGR2GRAY)
    mean, std = np.mean(gray), np.std(gray)
    Image_std_reduced = (gray - mean) / std
    img_red_bruit = restoration.denoise_nl_means(Image_std_reduced)
    img_red_bruit = np.expand_dims(img_red_bruit, axis=-1)
    img_red_bruit = np.repeat(img_red_bruit, 3, axis=-1)
    img_red_bruit = np.expand_dims(img_red_bruit, axis=0)
    return img_red_bruit

st.set_page_config(page_title="Classification des races de chien", page_icon=":camera:", layout="wide")


st.title("Classification des races de chien")
file = st.file_uploader("Veuiller glisser ou selectionner une image", type=["jpg", "jpeg", "png"])
if file is not None:
    image = Image.open(file)
    st.image(image)
#if file:
    #image=imread(str(file))
    processed_image = process_image(image)
    if processed_image is not None:
        # Prédire la classe de l'image modifiée en utilisant le modèle
        prediction = model_final.predict(processed_image)
        race_pred = np.argmax(prediction, axis=1)[0]
        proba_race = round(100*prediction[0][race_pred])
        race = Races[race_pred]

        st.write("Il y a " + str(proba_race) + "% de chance qu'il s'agisse d'un " + race)
