import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
import os
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
from PIL import Image

import keras
from keras.datasets import mnist
import torch
import torch.nn.functional as F
import random

from PIL import Image


st.set_page_config(page_title="Denoising Ai",
                   page_icon="📸",
                   )


def main():
    models()


def models():
    st.title("Image Denoising using Auto Encoders")
    st.subheader(
        'You can predict on sample images or you can upload a noisy image and get its denoised output.')
    st.subheader(
        'NOTE: Please upload an image with size of (28, 28)')

    selection = st.selectbox("Choose how to load image", [
                             "<Select>", "Upload an Image", "Predict on sample Images"])

    if selection == "Upload an Image":
        image = st.file_uploader('Upload the image below')
        predict_button = st.button('Predict on uploaded image')
        if predict_button:
            if image is not None:
                file_bytes = np.asarray(
                    bytearray(image.read()), dtype=np.uint8)
                nsy_img = cv2.imdecode(file_bytes, 1)
                prediction(nsy_img)
            else:
                st.text('Please upload the image')

    if selection == 'Predict on sample Images':
        option = st.selectbox('Select a sample image', ('<select>', 'Mnist'))
        if option == '<select>':
            pass
        else:
            path = os.path.join(os.getcwd(), 'NoisyImage/')
            nsy_img = cv2.imread(path+option+'.png')
            model_selector(nsy_img)


def get_model():
    trained_model = tf.keras.models.load_model(
        f'./models/autoencoder_mnist.h5')

    return trained_model


def model_selector(nsy_img):
    st.write('The selected model is Auto encoder')

    btn_status = st.button('Train the model')
    if btn_status:
        prediction(nsy_img)


def prediction(img):
    state = st.text(
        f'\n Please wait while Auto encoder denoise the image.....')
    progress_bar = st.progress(0)
    start = time.time()

    # Load the data
    (x_train, _), (x_test, _) = mnist.load_data()

    # Importing data
    num_data = 1000
    frac_train = 0.7
    frac_test = 0.3
    x_train = x_train[0:int(frac_train*num_data)]
    x_test = x_test[0:int(frac_test*num_data)]
    dimension = x_test.shape[1]

    # Pre-processing data
    norm_factor = 255.
    x_test = x_test.astype('float32')/norm_factor
    x_test = np.reshape(x_test, (len(x_test), dimension, dimension, 1))

    progress_bar.progress(30)

    # Add noise to the images
    noise_factor = 0.5
    x_test_noisy = x_test + noise_factor * \
        np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    x_test_noisy = np.reshape(
        x_test_noisy, (len(x_test_noisy), dimension, dimension, 1))

    # Building the Autoencoder
    model = get_model()
    pred_img = model.predict(np.squeeze(x_test_noisy))

    progress_bar.progress(70)
    end = time.time()

    # TODO: Select random image
    index = random.randint(0, len(x_test_noisy))

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(np.squeeze(x_test_noisy[index]))
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].title.set_text("Noisy Image")

    ax[1].imshow(np.squeeze(np.squeeze(pred_img[index])))
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].title.set_text("Predicted Image")

    st.pyplot(fig)

    noisy_image = np.squeeze(x_test_noisy[index])
    pred_image = np.squeeze(pred_img[index])

    img1 = Image.fromarray((noisy_image).astype(np.uint8))
    img2 = Image.fromarray((pred_image).astype(np.uint8))

    print(x_test_noisy[index].shape,
          pred_img[index].shape)

    image_compare(img1, img2)

    progress_bar.progress(100)
    st.write('Time taken for prediction :',
             str(round(end-start, 3))+' seconds')
    progress_bar.empty()
    state.text('\n Completed!')


def image_compare(img1, img2):
    # Streamlit Image-Comparison Component Example
    # https://github.com/fcakyon/streamlit-image-comparison
    import streamlit as st
    from streamlit_image_comparison import image_comparison

    # render image-comparison
    image_comparison(
        img1=img1,
        img2=img2,
    )


if __name__ == "__main__":
    main()
