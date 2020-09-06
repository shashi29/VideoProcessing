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

#boto3 package
import boto3


textract = boto3.client('textract',region_name='us-east-1',aws_access_key_id="AKIAJCHTK5KUJIVKTTVA",aws_secret_access_key= "tt7HwkHd3jF+hYaGlRwq3Z9+pd2zcFJsJdEBz95y")

class StartVideo(object):
    """ Threading example class
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, img, subtitle_path = None, interval=1):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.interval = interval
        self.img = img
        self.subtitle_path = subtitle_path
        thread = threading.Thread(target=self.extract_text_frame, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()  

    def extract_text_frame(self):
        #Crop the bottom part
        #run tesseract on top of that
        height, width, channel = self.img.shape
        crop_img = self.img[int(6/8*height):height,0:width]
        # Call Amazon Textract
        lower = np.array([0,0,205])
        upper = np.array([179,255,255])
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        crop_img = cv2.bitwise_and(crop_img,crop_img, mask= mask)
        img_name = 'temp.png'
        cv2.imwrite(img_name, crop_img)
        with open(img_name, 'rb') as document:
            imageBytes = bytearray(document.read())
            
        response = textract.detect_document_text(Document={'Bytes': imageBytes})
        
        final_str = ''
        # Print detected text
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                print ('\033[94m' +  item["Text"] + '\033[0m')
                final_str +=  item["Text"] 
                final_str += " "
        
        file_object = open('temp1.txt','a+')
        print(final_str)
        file_object.write(final_str)
        file_object.close()        

def processText(sentences):
    stop_words = set(stopwords.words('english'))
    sentences = re.sub('[^A-Za-z0-9 ]+', '', sentences)
    
    word_tokens = word_tokenize(sentences)
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    word_count = Counter(filtered_sentence)    
    #print(word_count)
    
    #Specific word counter 
    word_list = ['sofa', 'talk','cozy']
    
    filtered_word_dict = list()
    for key in word_count:
        if key in word_list:
            #print(key, word_count[key])
            filtered_word_dict.append([key,word_count[key]])
    #print(filtered_word_dict)
    return word_count.most_common(10),filtered_word_dict

def PlayVideo(video_path): 
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    count = 0 
    while True:
        count = count + 1
        grabbed, frame=video.read()
        #audio_frame, val = player.get_frame()
        
        if count % 10 == 0:
            runVideo = StartVideo(frame)
        try:
            # Reading data from file2 
            with open('temp1.txt') as fp2: 
                data = fp2.read() 
            word_count,word_count1 = processText(data)
            for indx,(key,value) in enumerate(word_count1):
                content = key + ' ' + str(value) 
                #content = '*****'
                indx = indx + 1
                frame = cv2.putText(frame, content, (50,indx*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

        except Exception as ex:
            continue

        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        #if val != 'eof' and audio_frame is not None:
            #audio
            #img, t = audio_frame

        fp2.close()
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    video_path = "1.mp4"
    audio_path = "audio.wav"
    encoding = 'LINEAR16'
    lang ='hi-IN'
    file = open('temp1.txt','w')
    file.close()
    #Extract audio from video using ffmpeg
    runVideo = PlayVideo(video_path)