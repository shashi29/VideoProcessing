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

#Image comparision
from skimage import measure
from PIL import Image
from measure_img_similarity import *
#http://sebastiandahlgren.se/2014/06/27/running-a-method-as-a-background-thread-in-python/
#https://gist.github.com/sebdah/832219525541e059aefa

def sift_sim(img_a, img_b):
  '''
  Use SIFT features to measure image similarity
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  # initialize the sift feature detector
  orb = cv2.ORB_create()

  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)

  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # match.distance is a float between {0:100} - lower means more similar
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 70]
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)

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
 
#Convert numpy image to the io format       
#https://jdhao.github.io/2019/07/06/python_opencv_pil_image_to_bytes/
def detect_text_googleVisonApi(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    #with io.open(path, 'rb') as image_file:
    #    content = image_file.read()
    is_success, im_buf_arr = cv2.imencode(".jpg", path)
    content = im_buf_arr.tobytes()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


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
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
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
        
        try:
            if len(crop_list) > 2:
                prev_img = crop_list[-1]
                boxA,crop_img = find_text_region_crop(img)
                boxB,_ = find_text_region_crop(prev_img)
                iou = bb_intersection_over_union(boxA, boxB)
                if iou < 1.0:
                    #cv2.imshow("prev_img", prev_img)
                    #cv2.imshow("current_img",img)
                    #cv2.destroyAllWindows()
                    #th = threading.Thread(target=extract_text, args=(img, detector, recognizer,))
                    #th = threading.Thread(target=extract_text_frame, args=(crop_img,))
                    
                    th = threading.Thread(target=detect_text_googleVisonApi, args=(crop_img,))
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

    video_path = "5.mp4"
    audio_path = "audio.wav"
    encoding = 'LINEAR16'
    lang ='hi-IN'
    subtitle_path = "subtitle.txt"
    file = open('temp1.txt','w')
    file.close()
    #Extract audio from video using ffmpeg
    runVideo = PlayVideo(video_path, detector, recognizer)
    
