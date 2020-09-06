import pandas as pd
import os 
import cv2
import numpy as np
import re
import sys
import glob
from collections import Counter 
from ffpyplayer.player import MediaPlayer

#Tokenize and remove stop words
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords

#Google speech to text
import wave
from typing import Tuple
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

#Threading
import threading
import time
import timeit

# Import the AudioSegment class for processing audio and the 
# split_on_silence function for separating out silent chunks.
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.playback import play
from playsound import playsound

#Not working package
#import pyaudio

#Paramters for the audio
chunk = 1024  

# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def create_audio_chunks(sound):
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")

def playAudio(audio):
    #Unable to install pyaudio 
    f = wave.open(audio,"rb")  
    #instantiate PyAudio  
    p = pyaudio.PyAudio()  
    #open stream  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = f.getframerate(),  
                    output = True)  
    #read data  
    data = f.readframes(chunk)  

    #play stream  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  

    #stop stream  
    stream.stop_stream()  
    stream.close()  

    #close PyAudio  
    p.terminate()

if __name__ == "__main__":
    video_path = sys.argv[-1]
    audio_path = "audio.wav"
    audio_chunk_folder = "audio-chunks"
    #command = f"ffmpeg -i {video_path} -acodec pcm_s16le -ac 1 {audio_path}"
    #os.system(command)
    audio = AudioSegment.from_wav(audio_path)
    #create_audio_chunks(audio)

    #One for parallel thread for audio word timestamps
    

    #Another function to mute the audio for the given word

    #This will be parallel thread of playing audio chunk
    for chunk_audio in glob.glob(f"{audio_chunk_folder}/*"):
        playsound(chunk_audio)
    
    #Parallel thread for displaying the video
    #playVideo(video_path)
