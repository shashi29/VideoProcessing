"""
Created on Fri Sep 18 23:02:57 2020

@author: shashi.raj
"""

import cv2
import numpy
import sys
import os
import io

from collections import Counter 
from ffpyplayer.player import MediaPlayer

#Google API
import wave
from typing import Tuple
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.cloud import vision

#Threading
import threading
import time

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import time

from utility import *

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="credentials.json"

def detect_text_googleVisonApi(path, frame_count):
    with open('mask_word.txt') as fp1: 
        mask_word = fp1.read() 
            
    mask_word = mask_word.split("\n")
    client = vision.ImageAnnotatorClient()

    is_success, im_buf_arr = cv2.imencode(".jpg", path)
    content = im_buf_arr.tobytes()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    file_name = f'intermediate/temp_{frame_count}.txt'
    file = open(file_name,'w')

    for text in texts:
        if text.description in mask_word:            
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
    print(f"[INFO] Processing Frame {frame_count}")


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
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    crop_img = img[y:y+h,x:x+w]
    return [x,y,x+w,y+h],crop_img


def extract_mask_bbox_info(video_path): 
    video=cv2.VideoCapture(video_path)
    count = 0
    crop_list = list()
    while (video.isOpened()):
        count = count + 1
        grabbed, frame=video.read()
        if not grabbed:
            print("[INFO] End of video")
            break
        
        height, width, channel = frame.shape
        lower_height = int(6/8*height)
        img = frame[lower_height:height,0:width]
        
            
        if len(crop_list) > 1:
            prev_frame = crop_list[-1]
            prev_img = prev_frame[lower_height:height,0:width]
            boxA, crop_imgA = find_text_region_crop(img)
            boxB, crop_imgB = find_text_region_crop(prev_img)            
            iou = bb_intersection_over_union(boxA, boxB)
            if iou < 1.0:
                th = threading.Thread(target = detect_text_googleVisonApi, args=(img, count, ))
                th.daemon = True
                th.start()
        crop_list.append(frame)


def playVideo(video_path):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    count = -1
    previous_processed_frames = list()
    previous_processed_frames.append(0)     #Hack for first frame
    while (video.isOpened()):
        count = count + 1
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("[INFO] End of video")
            break
                    
        height, width, channel = frame.shape
        Offset = int(6/8*height)
        print(f"[INFO] Frame {count}")
        file_name = f'intermediate/temp_{count}.txt'
        if os.path.isfile(file_name):
            previous_processed_frames.append(count)
            print(f"[INFO] Processing Frames {count}")
            with open(file_name) as fp1: 
                mask_word_box = fp1.read() 
            
            mask_word_box = mask_word_box.split("\n")
            mask_word_box.pop()
            if len(mask_word_box) > 0:
                for bb_mask in mask_word_box:
                    bb_mask = bb_mask.split(" ")
                    cv2.rectangle(frame,(int(bb_mask[0]),int(bb_mask[1])+Offset),(int(bb_mask[2]),int(bb_mask[3])+Offset),(0,0,255),-1)
        fp1.close()

        if os.path.isfile(file_name) == False:
            prev_count = previous_processed_frames[len(previous_processed_frames)-1]
            file_name = f"intermediate/temp_{prev_count}.txt"
            if os.path.isfile(file_name):
                with open(file_name) as fp1: 
                    mask_word_box = fp1.read() 
                
                mask_word_box = mask_word_box.split("\n")
                mask_word_box.pop()
                if len(mask_word_box) > 0:
                    for bb_mask in mask_word_box:
                        bb_mask = bb_mask.split(" ")
                        cv2.rectangle(frame,(int(bb_mask[0]),int(bb_mask[1])+Offset),(int(bb_mask[2]),int(bb_mask[3])+Offset),(0,0,255),-1)
            fp1.close()

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        cv2.imshow("Video", frame)
    video.release()
    cv2.destroyAllWindows()

def word_timestamp(response):
    start_time = list()
    end_time = list()
    with open('mask_word.txt') as fp1: 
        mask_word = fp1.read() 
            
    bad_word = mask_word.split("\n")
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
    

if __name__ == "__main__":

    video_path = "5.mp4"
    audio_path = "audio.wav"
    BUCKET_NAME = "audio_2020"

    os.remove(audio_path)
    #th = threading.Thread(target = extract_mask_bbox_info, args=(video_path, ))
    #th.daemon = True
    #th.start()
    print("[INFO] Find details of all mask word in video")
    file = open('intermediate/temp_0.txt','w')
    file.close()
        
    channels, bit_rate, sample_rate = video_info(video_path)
    blob_name = video_to_audio(video_path, audio_path, channels, bit_rate, sample_rate)

    gcs_uri = f"gs://{BUCKET_NAME}/{audio_path}"
    response = long_running_recognize(gcs_uri, channels, sample_rate)

    start_time, end_time = word_timestamp(response)

    #Now start the live vide with the thread
    unmute_audio()
    st_time = time.time()
    print("[INFO] Starting video")
    startDemo = threading.Thread(target = playVideo, args=(video_path, ))
    startDemo.daemon = True
    startDemo.start()
    
    while True:
        time_dt = time.time() - st_time
        time_dt = int(time_dt)
        if time_dt in start_time:
            mute_audio()
            time.sleep(0.5)
            unmute_audio()
    
    
    
    