import pandas as pd
import numpy as np
import cv2
import os

video_path = "newVideo.mp4"
video_srt_path = "newVideo_srt.mp4"

cap = cv2.VideoCapture(video_path)
cap_srt = cv2.VideoCapture(video_srt_path)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640,480))

while(True):
    
    ret, frame = cap.read()
    ret, frame_srt = cap_srt.read()
    
    height, width, channel = frame_srt.shape
    lower_height = int(9/10 * height)
    img = frame_srt[lower_height : height, 0 : width]
    frame[lower_height : height, 0 : width] = img
    cv2.imshow('frame', frame)
    cv2.imshow('frame_srt', frame_srt)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap_srt.release()
out.release()
cv2.destroyAllWindows()