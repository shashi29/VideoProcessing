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

#NEW OCR
from src.detector import Detector
from src.recognizer import Recognizer
import os
import sys
import cv2

from skimage import measure

#http://sebastiandahlgren.se/2014/06/27/running-a-method-as-a-background-thread-in-python/
#https://gist.github.com/sebdah/832219525541e059aefa

def extract_text_frame(img):
    #Crop the bottom part
    #run tesseract on top of that
    height, width, channel = img.shape
    img = img[int(6/8*height):height,0:width]
    #lower = np.array([0,0,205])
    #upper = np.array([179,255,255])
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #mask = cv2.inRange(hsv, lower, upper)
    #res = cv2.bitwise_and(img,img, mask= mask)
    cv2.imwrite("temp.jpg",img)
    custom_config = r'--oem 3 --psm 6'
    #text = pytesseract.image_to_data(img, config=custom_config)
    text = pytesseract.image_to_string(img, lang='eng', \
        config='--psm 6 --oem 3 -c tessedit_char_blacklist=0123456789')
    text = re.sub('[^A-Za-z0-9 ]+', ' ', text)
    print(text)

def extract_text(img, detector, recognizer):
    full_text = ''
    file_object = open('temp1.txt','a+')
    roi, _, _, _ = detector.process(img)
    for i, img in enumerate(roi):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        text, _, _ = recognizer.process(gray)
        #print(transcript)
        text = text + ' '
        file_object.write(text)
    file_object.close()        
        

def processText(sentences):
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

def PlayVideo(video_path, detector, recognizer): 
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    count = 0
    crop_list = list()
    while (video.isOpened()):
        count = count + 1
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        
        height, width, channel = frame.shape
        img = frame[int(6/8*height):height,0:width]
        
        if count % 5 == 0:
            try:
                prev_img = crop_list[-1]
                s = measure.compare_ssim(img, prev_img,multichannel=True)
                
                if s < 0.9:
                    th = threading.Thread(target=extract_text, args=(img, detector, recognizer,))
                    th.daemon = True
                    th.start()                    
                crop_list.append(img)
            except Exception as ex:
                crop_list.append(img)
                continue

        with open('temp1.txt') as fp1: 
            data = fp1.read() 
            
        word_count,word_count1 = processText(data)
        for indx,(key,value) in enumerate(word_count):
            content = key + ' ' + str(value) 
            indx = indx + 1
            frame = cv2.putText(frame, content, (50,indx*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
 

        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame

        
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    detector = Detector()
    detector.load()
    
    recognizer = Recognizer()
    recognizer.load()    

    video_path = "1.mp4"
    audio_path = "audio.wav"
    encoding = 'LINEAR16'
    lang ='hi-IN'
    subtitle_path = "subtitle.txt"
    #Extract audio from video using ffmpeg
    runVideo = PlayVideo(video_path, detector, recognizer)
    
    