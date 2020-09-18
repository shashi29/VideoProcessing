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
from google.cloud import vision
import io

#Image comparision
from skimage import measure
from PIL import Image
from measure_img_similarity import *
#http://sebastiandahlgren.se/2014/06/27/running-a-method-as-a-background-thread-in-python/
#https://gist.github.com/sebdah/832219525541e059aefa

def extract_text_frame(img):
    #Crop the bottom part
    #run tesseract on top of that
    height, width, channel = img.shape
    img = img[int(6/8*height):height,0:width]
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


#Convert numpy image to the io format       
#https://jdhao.github.io/2019/07/06/python_opencv_pil_image_to_bytes/
def detect_text_googleVisonApi(path):
    """Detects text in the file."""
    
    #Read mask word list file
    with open('mask_word.txt') as fp1: 
        mask_word = fp1.read() 
    
        
    mask_word = mask_word.split("\n")
    client = vision.ImageAnnotatorClient()

    is_success, im_buf_arr = cv2.imencode(".jpg", path)
    content = im_buf_arr.tobytes()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    file = open('temp2.txt','w')

    for text in texts:
        if text.description in mask_word:            

            #print('\n"{}"'.format(text.description))            
            #vertices = (['[{},{}]'.format(vertex.x, vertex.y)
            #            for vertex in text.bounding_poly.vertices])
            x_list = list()
            y_list = list()
            for vertex in text.bounding_poly.vertices:
                x_list.append(vertex.x)
                y_list.append(vertex.y)
            xmin = min(x_list)
            ymin = min(y_list)
            xmax = max(x_list)
            ymax = max(y_list)
            
            file.write(f'{xmin} {ymin} {xmax} {ymax}\n')

    file.close()
    fp1.close()

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

#Process on crop img part to find the text containing area
#Then we will calcualte IOU between theose frames
#On the basis of that we will chage these
#https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/
def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def find_text_region_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                     cv2.CHAIN_APPROX_NONE) 
    #Find the area the index of largest
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    crop_img = img[y:y+h,x:x+w]
    return [x,y,x+w,y+h],crop_img

def PlayVideo(video_path, detector=None, recognizer=None): 
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    count = 0
    crop_list = list()
    while (video.isOpened()):
        count = count + 1
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        
        height, width, channel = frame.shape
        lower_height = int(8/10*height)
        img = frame[lower_height:height,0:width]
        
        #if count == 0:
            #First frame
            #detect_text_googleVisonApi(img)
            
        if len(crop_list) > 1:
            
            prev_frame = crop_list[-1]
            prev_img = prev_frame[lower_height:height,0:width]
            boxA, crop_imgA = find_text_region_crop(img)
            boxB, crop_imgB = find_text_region_crop(prev_img)            
            iou = bb_intersection_over_union(boxA, boxB)
            if iou < 0.99:
                #cv2.imshow("prev_img", crop_imgB)
                #cv2.imshow("current_img", crop_imgA)
                #th = threading.Thread(target = detect_text_googleVisonApiV2, args=(img, prev_img, ))
                #th.daemon = True
                #th.start()
                 
                #Plan run google vision api on both the images
                #If content is same , then clean 
                #cv2.destroyAllWindows()
                #th = threading.Thread(target=extract_text, args=(img, detector, recognizer,))
                #th = threading.Thread(target=extract_text_frame, args=(crop_img,))                
                #th = threading.Thread(target=detect_text_googleVisonApi, args=(img,))
                #th.daemon = True
                #th.start()

        crop_list.append(frame)

        with open('temp1.txt') as fp1: 
            data = fp1.read() 
            
        word_count,word_count1 = processText(data)
        for indx,(key,value) in enumerate(word_count):
            content = key + ' ' + str(value) 
            indx = indx + 1
            frame = cv2.putText(frame, content, (50,indx*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
 
        #Now read the content of file and mask the word
        with open('temp2.txt') as fp1: 
            mask_word_box = fp1.read() 
        
        mask_word_box = mask_word_box.split("\n")
        mask_word_box.pop()
        if len(mask_word_box) > 0:
            for bb_mask in mask_word_box:
                bb_mask = bb_mask.split(" ")
                cv2.rectangle(frame,(int(bb_mask[0]),int(bb_mask[1])+lower_height),(int(bb_mask[2]),int(bb_mask[3])+lower_height),(0,0,255),-1)


        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        if count > 1:
            cv2.imshow("Video", crop_list[-1])
            
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    #detector = Detector()
    #detector.load()
    
    #recognizer = Recognizer()
    #recognizer.load()    

    video_path = "5.mp4"
    audio_path = "audio.wav"
    encoding = 'LINEAR16'
    lang ='hi-IN'
    subtitle_path = "subtitle.txt"
    file = open('temp1.txt','w')
    file.close()
    file = open('temp2.txt','w')
    file.close()
    #Extract audio from video using ffmpeg
    runVideo = PlayVideo(video_path)
        