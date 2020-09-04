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

    def PlayVideo(self): 
        video=cv2.VideoCapture(self.video_path)
        player = MediaPlayer(self.video_path)
        while True:
            grabbed, frame=video.read()
            audio_frame, val = player.get_frame()
            with open('temp1.txt') as fp1: 
                data = fp1.read() 
              
            try:
                # Reading data from file2 
                with open('temp2.txt') as fp2: 
                    data2 = fp2.read() 
            except Exception as ex:
                #print(ex)
                continue
            # Merging 2 files 
            # To add the data of file2 
            # from next line 
            data += "\n"
            data += data2 
                        
            word_count,word_count1 = self.processText(data)
            for indx,(key,value) in enumerate(word_count):
                content = key + ' ' + str(value) 
                indx = indx + 1
                frame = cv2.putText(frame, content, (50,indx*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
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
            fp1.close()
            fp2.close()
        video.release()
        cv2.destroyAllWindows()

def read_wav_file(filename) -> Tuple[bytes, int]:
    with wave.open(filename, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)

    return buffer, rate

def simulate_stream(buffer: bytes, batch_size: int = 4096):
    buffer_len = len(buffer)
    offset = 0
    while offset < buffer_len:
        end_offset = offset + batch_size
        buf = buffer[offset:end_offset]
        yield buf
        offset = end_offset

def response_stream_processor(responses, file_object):

    print('interim results: ')

    transcript = ''
    whole_text = ''
    num_chars_printed = 0
    for indx,response in enumerate(responses):
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        if result.is_final:
            print(transcript)
            file_object.write(transcript)                
            whole_text += transcript
        #print('{0}final: {1}'.format(
        #    '' if result.is_final else 'not ',
        #    transcript
        #))
    return transcript

def google_streaming_stt(filename: str, lang: str, encoding: str):
    buffer, rate = read_wav_file(filename)

    client = speech.SpeechClient()

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding[encoding],
        sample_rate_hertz=rate,
        language_code=lang
    )

    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    audio_generator = simulate_stream(buffer)  # buffer chunk generator
    try:
        requests = (types.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
        print(requests)
        responses = client.streaming_recognize(streaming_config, requests)
        # Now, put the transcription responses to use.
    except Exception as ex:
        print(ex)
    for response in responses:
        try:
            result = response.results[0]
            transcript = result.alternatives[0].transcript
            if result.is_final:
                file_object = open('temp1.txt','a+')
                print(transcript)
                file_object.write(transcript)
                file_object.close()
            else:
                file_object = open('temp2.txt','w')
                print(transcript)
                file_object.write(transcript)
                file_object.close()
        except Exception as ex:
            #print(ex)
            continue
 

if __name__ == "__main__":
    
    
    link="https://www.youtube.com/watch?v=6M5VXKLf4D4"
    #video_path = download_video(link)
    video_path = "video6.mp4    "
    audio_path = "audio.wav"
    #video_path = r"D:\Extra\Demo\Big Runway Latent Walk Interpolation Interface Improvements! StyleGAN and BigGAN.mp4"
    channels, bit_rate, sample_rate = video_info(video_path)
    video_to_audio(video_path, audio_path, channels, bit_rate, sample_rate)
    encoding = 'LINEAR16'
    lang ='en-US'
    file = open('temp1.txt','w')
    file.close()
    file = open('temp2.txt','w')
    file.close()
    time.sleep(5)
    runVideo = StartVideo(video_path)
    google_streaming_stt(audio_path, lang, encoding)

    