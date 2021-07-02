import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import numpy as np
import soundfile as sf
import os
import librosa
import glob
from helper import draw_embed, create_spectrogram, read_audio, record, save_record, predict, predicitons_plot

# Declare the component:
my_component = components.declare_component("my_component",
    path="components/my_component/frontend/build")

# st.set_page_config(layout="wide")

st.set_page_config("Speech Emotion Detection", "./assets/img/unipi_logo.png")

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }}
    # header {{
    #     visibility: hidden;
    # }}
    # footer {{
    #     visibility: hidden;
    # }}
</style>
""",
        unsafe_allow_html=True,
    )

"# Speech emotion recognition"

st.header("1. Record your own voice")

# filename = st.text_input("Give a filename: ", value="sample")

recording = my_component()


if st.button(f"Click to analyze"):
    # @TODO check if sample.wav exists otherwise throw warning to record
    
    # if filename == "":
    #     st.warning("Choose a filename.")
    # else:
    # record_state = st.text("Recording...")
    # duration = 1  # seconds
    # fs = 22050
    # myrecording = record(duration, fs)
    # record_state.text(f"Saving sample as {filename}.wav")

    filename = 'sample'
    path_myrecording = f"./samples/{filename}.wav"

    # save_record(path_myrecording, myrecording, fs)
    # record_state.text(f"Done! Saved sample as {filename}.wav")

    st.audio(read_audio(path_myrecording))

    # Make prediction
    predictions = predict(path_myrecording)
    st.text(predictions)
    # Plot predictions
    fig = predicitons_plot(predictions.to_numpy()[0][:-1])
    st.pyplot(fig)

    fig = create_spectrogram(path_myrecording)
    st.pyplot(fig)