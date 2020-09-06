import pandas as pd
import os 
import cv2
import numpy as np
import re
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

#Tesseract
import pytesseract


#http://sebastiandahlgren.se/2014/06/27/running-a-method-as-a-background-thread-in-python/

def extract_text_frame(img):
    #Crop the bottom part
    #run tesseract on top of that
    height, width, channel = img.shape
    img = img[int(6/8*height):height,0:width]
    lower = np.array([0,0,205])
    upper = np.array([179,255,255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img,img, mask= mask)

    custom_config = r'--oem 3 --psm 6'
    cv2.imwrite("temp.jpg",res)
    try:
        print(pytesseract.image_to_string(res ))#, config=custom_config))
    except Exception as ex:
        print(ex)


def PlayVideo(video_path): 
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    count = 0 
    while True:
        count = count + 1
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        
        if count % 15 == 0:
            th = threading.Thread(target=extract_text_frame, args=(frame,))
            th.start()

        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(45) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame

        if count % 15 == 0:
            th.join()

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    video_path = "1.mp4"
    audio_path = "audio.wav"
    encoding = 'LINEAR16'
    lang ='hi-IN'
    subtitle_path = "subtitle.txt"
    #Extract audio from video using ffmpeg
    runVideo = PlayVideo(video_path)
    
