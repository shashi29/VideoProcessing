import os
import cv2
import numpy as np
from pydub.utils import mediainfo

def video_to_audio(video_filepath, audio_filename, video_channels, video_bit_rate, video_sample_rate):
    command = f"ffmpeg -i {video_filepath} -b:a {video_bit_rate} -ac {video_channels} -ar {video_sample_rate} -vn -y {audio_filename}"
    #subprocess.call(command, shell=True)
    os.system(command)
    
def video_info(video_filepath):
    video_data = mediainfo(video_filepath)
    channels = video_data["channels"]
    bit_rate = video_data["bit_rate"]
    sample_rate = video_data["sample_rate"]
    return channels, bit_rate, sample_rate

def playVideo(video_path):
    video=cv2.VideoCapture(video_path)
    count = -1
    previous_processed_frames = list()
    previous_processed_frames.append(0)     #Hack for first frame
    while (video.isOpened()):
        count = count + 1
        grabbed, frame=video.read()
        if not grabbed:
            print("[INFO] End of video")
            break
                    
        height, width, channel = frame.shape
        Offset = int(6/8*height)
        #print(f"[INFO] Frame {count}")
        file_name = f'intermediate/temp_{count}.txt'
        if os.path.isfile(file_name):
            previous_processed_frames.append(count)
            #print(f"[INFO] Processing Frames {count}")
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
    


video_path = r"C:/Users/shashi.raj/Downloads/output.mp4"

video = cv2.VideoCapture(video_path) 
if (video.isOpened() == False):  
    print("Error reading video file") 
frame_width = int(video.get(3)) 
frame_height = int(video.get(4))    
size = (frame_width, frame_height)
fps = video.get(cv2.CAP_PROP_FPS)
  
result = cv2.VideoWriter('filename.mp4',  
                         cv2.VideoWriter_fourcc(*'MP4V'), 
                         fps, size) 
    
while(True): 
    ret, frame = video.read() 
  
    if ret == True:  
        result.write(frame) 
    else: 
        break

video.release() 
result.release() 
cv2.destroyAllWindows() 
   
print("The video was successfully saved") 

channels, bit_rate, sample_rate = video_info(video_path)
audio_filename = "new_audio.wav"
video_to_audio(video_path, audio_filename, channels, bit_rate, sample_rate)

command = f"ffmpeg -i filename.mp4 -i new_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k -y newVideo.mp4"
os.system(command)
