import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data
import librosa
from pathlib import Path
import sounddevice as sd
import wavio
import numpy as np
import pandas as pd 
import soundfile as sf
from tensorflow.keras.models import model_from_json

# Create a configuration class to help if I want to change parameters later
class Config:
    def __init__(self, n_mfcc = 26, n_feat = 13, n_fft = 552, sr = 22050, window = 0.4, test_shift = 0.1):
        self.n_mfcc = n_mfcc
        self.n_feat = n_feat
        self.n_fft = n_fft
        self.sr = sr
        self.window = window
        self.step = int(sr * window)
        self.test_shift = test_shift
        self.shift = int(sr * test_shift)
        
config = Config()

def draw_embed(embed, name, which):
    """
    Draws an embedding.

    Parameters:
        embed (np.array): array of embedding

        name (str): title of plot


    Return:
        fig: matplotlib figure
    """
    fig, embed_ax = plt.subplots()
    # plot_embedding_as_heatmap(embed)
    embed_ax.set_title(name)
    embed_ax.set_aspect("equal", "datalim")
    embed_ax.set_xticks([])
    embed_ax.set_yticks([])
    embed_ax.figure.canvas.draw()
    return fig


def create_spectrogram(voice_sample):
    """
    Creates and saves a spectrogram plot for a sound sample.

    Parameters:
        voice_sample (str): path to sample of sound

    Return:
        fig
    """

    in_fpath = Path(voice_sample.replace('"', "").replace("'", ""))
    original_wav, sampling_rate = librosa.load(str(in_fpath))

    # Plot the signal read from wav file
    fig = plt.figure(figsize=(10,8))
    plt.subplot(211)
    plt.title(f"Spectrogram of file {voice_sample}")

    plt.plot(original_wav)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(212)
    plt.specgram(original_wav, Fs=sampling_rate)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # plt.savefig(voice_sample.split(".")[0] + "_spectogram.png")
    return fig

def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

def record(duration=5, fs=48000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording

def save_record(path_myrecording, myrecording, fs):
    with open(path_myrecording, 'w') as new_file:
        sf.write(path_myrecording, myrecording, fs)
    # wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    wav, sr = librosa.load(path_myrecording)
    wav = envelope(wav, sr, 0.05) # @TODO play with threshold
    with open(path_myrecording, 'w') as new_file:
        sf.write(path_myrecording, wav, fs)
    return None

######################

def envelope(y, sr, threshold):
    mask = []
    y_abs = pd.Series(y).apply(np.abs)
    y_mean = y_abs.rolling(window = int(sr/10), min_periods = 1, center = True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return np.array(y[mask])

# Model
def predict(audio_path):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")

    all_results = []
    local_results = []
    _min, _max = float('inf'), -float('inf')

    # Load the file
    wav, sr = librosa.load(audio_path)

    # Create an array to hold features for each window
    X = []

    # Iterate over sliding 0.4s windows of the audio file
    for i in range(int((wav.shape[0]/sr-config.window)/config.test_shift)):
        X_sample = wav[i*config.shift: i*config.shift + config.step] # slice out 0.4s window
        X_mfccs = librosa.feature.mfcc(X_sample, sr, n_mfcc = config.n_mfcc, n_fft = config.n_fft,
                                        hop_length = config.n_fft)[1:config.n_feat + 1] # generate mfccs from sample
        
        _min = min(np.amin(X_mfccs), _min)
        _max = max(np.amax(X_mfccs), _max) # check min and max values
        X.append(X_mfccs) # add features of window to X
    # Put window data into array, scale, then reshape
    X = np.array(X)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Feed data for each window into model for prediction
    for i in range(X.shape[0]):
        window = X[i].reshape(1, X.shape[1], X.shape[2], 1)
        local_results.append(loaded_model.predict(window))
    
    # Aggregate predictions for file into one then append to all_results
    local_results = (np.sum(np.array(local_results), axis = 0)/len(local_results))[0]
    local_results = list(local_results)
    # Final prediction
    prediction = np.argmax(local_results)
    labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
    local_results.append(labels[prediction])
    all_results.append(local_results)


    
    # Turn all results into a dataframe
    df_cols = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'prediction']
    all_results = pd.DataFrame(all_results, columns = df_cols)
    return all_results


def predicitons_plot(data):
    # freqs = [301, 96, 53, 81, 42]

    fig, ax = plt.subplots()

    labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
    # labels = ['neutral']*7
    p1 = ax.bar(np.arange(len(labels)), data, 0.8, color="yellow")
    plt.ylim(0, plt.ylim()[1]+0.5)

    for rect1, label in zip(p1, labels):
        arr_img = plt.imread(f'./assets/img/{label}_bob_min.png', format='png')
        # arr_img = plt.imread(f'./assets/img/fearful.png', format='png')
        imagebox = OffsetImage(arr_img, zoom=0.45)
        imagebox.image.axes = ax
        height = rect1.get_height()

        ab = AnnotationBbox(imagebox,
                            (rect1.get_x() + rect1.get_width()/2, height+0.12),
                            bboxprops=dict(edgecolor="white")
                            )

        ax.add_artist(ab)
    return fig