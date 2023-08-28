import streamlit as st
import os
import subprocess
import imageio

import tensorflow as tf
from tensorflow.keras.models import load_model

from utils import load_data, num_to_char
# from modelutil import load_model

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# # Set the XLA_FLAGS environment variable
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6"
# # Set the TF_DISABLE_JIT environment variable
# os.environ["TF_DISABLE_JIT"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set the layout to wide
st.set_page_config(layout='wide')

ffmpeg_path = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'

# Setup the sidebar
with st.sidebar:
    st.image("./assets/background.jpg")
    st.title('Scribot')
    st.info('This is a Lip reading application developed using LipNet deep learning model')
    
st.title("Scribot App")

st.image("./assets/background2.jpg")

# Generating a list of options or videos
options = os.listdir(os.path.join('.', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate wo columns
col1, col2 = st.columns(2)

if options:
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('.','data','s1', selected_video)
        # os.system(f'ffmpeg -i {file_path} -vcodec libx264 data/test_video.mp4 -y')
        subprocess.call([ffmpeg_path, '-i', './data/s1/bbaf2n.mpg', '-vcodec', 'libx264', './data/test_video.mp4', '-y'])
        
        # rendering the video inside the app
        video = open('./data/test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)
    
    with col2:
        st.info('What the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        
        # # Convert the list of frames to a NumPy array
        # video_array = tf.stack(video)  # Stack the frames to create a tensor
        # video_array = video_array.numpy()  # Convert the tensor to a NumPy array
        # imageio.mimsave('animation.gif', video_array, fps=10)
        
        # st.image('animation.gif', width=400) 

        st.info('The output of the machine learning model as tokens')
        model = load_model('./model/lipReadingModel.h5')
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        