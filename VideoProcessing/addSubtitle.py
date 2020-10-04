import os
import requests
import cv2
import numpy as np
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
from utility import *
from run import *




def add_srt_video(srt_path, video_path):
    print("[INFO] Adding srt file to the final video")
    result_video_name = video_path[:-4] + '_result.mp4'
    generator = lambda txt: TextClip(txt, font='Arial', fontsize=16, color='white')
    subtitles = SubtitlesClip(srt_path, generator)
    video = VideoFileClip(video_path)
    result = CompositeVideoClip([video, subtitles.set_pos(('center','bottom'))])
    result.write_videofile(video_path, fps=video.fps, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
        
    UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
    DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
    ALLOWED_EXTENSIONS = {'mp4'}
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    video_path = r"C:/Users/shashi.raj/Desktop/Projects/VideoProcessing/test_video/test3.mp4"
    video_name = os.path.basename(video_path)
    video_name = video_name.split(".")[0]
    raw_audio_name = f'{video_name}_audio.wav'
    beep_path = "beep.wav"
    raw_audio_path = os.path.join(UPLOAD_FOLDER, raw_audio_name)
    processed_audio_name = f'{video_name}_final.wav'
    processed_audio_path = os.path.join(UPLOAD_FOLDER, processed_audio_name)
    BUCKET_NAME = "audio_2020"
    no_audio_video_path = video_path[:-4] + '_No_Audio.mp4'
    final_video = video_name + '_final_result.mp4'
    processed_video = os.path.join(DOWNLOAD_FOLDER, final_video)
    
    channels, bit_rate, sample_rate = video_info(video_path)
    blob_name = video_to_audio(video_path, raw_audio_path, channels, bit_rate, sample_rate)
    
    gcs_uri = f"gs://{BUCKET_NAME}/{raw_audio_name}"
    response = long_running_recognize(gcs_uri, channels, sample_rate)
    srt = subtitle_generation(response)
    with open("subtitles.srt", "w") as f:
        f.write(srt)
    
    response_df = word_timestamp(response)
    
    #mask audio
    mask_audio = process_audio(raw_audio_path, beep_path, response_df)
    mask_audio.export(processed_audio_path, format="wav")
    #Remove audio 
    command = f"ffmpeg -i {video_path} -vcodec copy -an -y {no_audio_video_path}"
    os.system(command)
    command = f"ffmpeg -i {no_audio_video_path} -i {processed_audio_path} -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k -y {processed_video}"
    os.system(command)
    #add_srt_video("subtitles.srt", processed_video)
    