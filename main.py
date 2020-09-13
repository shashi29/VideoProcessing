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
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import time

from utility import *

class StartVideo(object):
    """ Threading example class
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, video_path, subtitle_path = None, interval=1):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.interval = interval
        self.video_path = video_path
        self.subtitle_path = subtitle_path
        thread = threading.Thread(target=self.PlayVideo, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def processText(self, sentences):
        stop_words = set(stopwords.words('english'))
        sentences = re.sub('[^A-Za-z0-9 ]+', '', sentences)
        word_tokens = word_tokenize(sentences)
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        word_count = Counter(filtered_sentence)    
        #print(word_count)
        
        #Specific word counter 
        word_list = ['deep','learning','AI','amazing']
        
        filtered_word_dict = list()
        for key in word_count:
            if key in word_list:
                #print(key, word_count[key])
                filtered_word_dict.append([key,word_count[key]])
        #print(filtered_word_dict)
        return word_count.most_common(10),filtered_word_dict

    def extract_text_frame(self,img):
        #Crop the bottom part
        #run tesseract on top of that
        print("I am here")
        custom_config = r'--oem 3 --psm 6'
        try:
            print(pytesseract.image_to_string(img, config=custom_config))
        except Exception as ex:
            print(ex)

    def PlayVideo(self): 
        video=cv2.VideoCapture(self.video_path)
        player = MediaPlayer(self.video_path)

        while True:
            grabbed, frame=video.read()
            audio_frame, val = player.get_frame()
            if not grabbed:
                print("End of video")
                break
            if cv2.waitKey(28) & 0xFF == ord("q"):
                break
            cv2.imshow("Video", frame)
            if val != 'eof' and audio_frame is not None:
                #audio
                img, t = audio_frame
                #print(img, t)
            #fileText.close()
        video.release()
        cv2.destroyAllWindows()


def mute_audio():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    currentVolumeDb = volume.GetMasterVolumeLevel()
    volume.SetMasterVolumeLevel(currentVolumeDb - 60.0, None)

def unmute_audio():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    currentVolumeDb = volume.GetMasterVolumeLevel()
    volume.SetMasterVolumeLevel(0.0, None)

#audio = AudioSegment.from_wav(audio_path)
#Idea 1
#create_audio_chunks(audio)
#One for parallel thread for audio word timestamps
#Another function to mute the audio for the given word
#This will be parallel thread of playing audio chunk    
#Parallel thread for displaying the video
#playVideo(video_path)

#Idea 2

def word_timestamp(response):
    start_time = list()
    end_time = list()
    bad_word = ['deep','learning','transformed','internet']
    for result in response.results:
        try:
            if result.alternatives[0].words[0].start_time.seconds:
                # bin start -> for first word of result
                start_sec = result.alternatives[0].words[0].start_time.seconds
                last_word_end_sec = result.alternatives[0].words[-1].end_time.seconds
                #print("Starting and end time for sentence",start_sec,last_word_end_sec)
        except Exception as ex:
            print(ex)
            
        #We have 30 second bins now we need to iterate over each save the info
        for i in range(len(result.alternatives[0].words) - 1):
            try:
                word = result.alternatives[0].words[i + 1].word
                word_start_sec = result.alternatives[0].words[i + 1].start_time.seconds
                word_end_sec = result.alternatives[0].words[i + 1].end_time.seconds
                if word in bad_word:
                    print(word,word_start_sec,word_end_sec)
                    start_time.append(word_start_sec)
                end_time.append(word_end_sec)
            except Exception as ex:
                print(ex)
    
    return start_time, end_time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="credentials.json"


#if __name__ == "__main__":
video_path = "1.mp4"
audio_path = "audio.wav"
BUCKET_NAME = "audio_2020"

#Task 1 : Mask the subtitle
#Task 2 : Mute the audio for that word

#channels, bit_rate, sample_rate = video_info(video_path)
#blob_name = video_to_audio(video_path, audio_path, channels, bit_rate, sample_rate)

#gcs_uri = f"gs://{BUCKET_NAME}/{audio_path}"
#response = long_running_recognize(gcs_uri, channels, sample_rate)

start_time, end_time = word_timestamp(response)

#Now start the live vide with the thread
unmute_audio()
st_time = time.time()
runVideo = StartVideo(video_path)
while True:
    time_dt = time.time() - st_time
    time_dt = int(time_dt)
    if time_dt in start_time:
        mute_audio()
        time.sleep(0.5)
        unmute_audio()
        
