import streamlit as st
from pathlib import Path
import numpy as np
import soundfile as sf
import os
import librosa
import glob
from helper import draw_embed, create_spectrogram, read_audio, record, save_record, predict, predicitons_plot

# st.set_page_config(layout="wide")

st.set_page_config("Speech emotion detection", "./assets/img/unipi_logo.png")

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

filename = st.text_input("Give a filename: ", value="sample")

if st.button(f"Click to Record"):
    if filename == "":
        st.warning("Choose a filename.")
    else:
        record_state = st.text("Recording...")
        duration = 5  # seconds
        fs = 22050
        myrecording = record(duration, fs)
        record_state.text(f"Saving sample as {filename}.wav")

        path_myrecording = f"./samples/{filename}.wav"

        save_record(path_myrecording, myrecording, fs)
        record_state.text(f"Done! Saved sample as {filename}.wav")

        st.audio(read_audio(path_myrecording))

        # Make prediction
        predictions = predict(path_myrecording)
        st.text(predictions)
        # Plot predictions
        fig = predicitons_plot(predictions.to_numpy()[0][:-1])
        st.pyplot(fig)

        fig = create_spectrogram(path_myrecording)
        st.pyplot(fig)