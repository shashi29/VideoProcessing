import pytube
import os
from google.cloud import storage
import json
import io
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
from google.cloud.speech_v1 import types
import subprocess
from pydub.utils import mediainfo
import subprocess
import math
import datetime
import srt
import wave
import sys
import os, sys, re, unicodedata

# custom imports
from gensim.utils import deaccent
from collections import Counter

verbose = True

#https://github.com/Ankur3107/nlp_preprocessing/blob/master/nlp_preprocessing/clean.py

def download_video(link):
    try: 
        #object creation using YouTube which was imported in the beginning 
        yt = pytube.YouTube(link) 
    except: 
        print("Connection Error") #to handle exception 
    video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    
    # rename the path
    new_path = video_path.split('/')
    new_filename = f"video.mp4"
    new_path[-1]= new_filename
    new_path='/'.join(new_path)
    os.rename(video_path, new_path)
        
    return new_path

def video_info(video_filepath):
    """ this function returns number of channels, bit rate, and sample rate of the video"""

    video_data = mediainfo(video_filepath)
    channels = video_data["channels"]
    bit_rate = video_data["bit_rate"]
    sample_rate = video_data["sample_rate"]

    return channels, bit_rate, sample_rate

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    #destination_blob_name = "audio.wav"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    #destination_blob_name = os.path.basename(destination_blob_name)
    print(destination_blob_name)
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("-------------------------------------------------")    
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def video_to_audio(video_filepath, audio_filename, video_channels, video_bit_rate, video_sample_rate):
    command = f"ffmpeg -i {video_filepath} -b:a {video_bit_rate} -ac {video_channels} -ar {video_sample_rate} -vn -y {audio_filename}"
    #subprocess.call(command, shell=True)
    os.system(command)

    #audio_filename = os.path.basename(audio_filename)
    blob_name = os.path.basename(audio_filename)
    BUCKET_NAME = "audio_2020"
    upload_blob(BUCKET_NAME, audio_filename, blob_name)
    return blob_name    

def long_running_recognize(storage_uri, channels, sample_rate):
    
    client = speech_v1.SpeechClient()

    config = {
        "language_code": "en-US",
        "sample_rate_hertz": int(sample_rate),
        "encoding": enums.RecognitionConfig.AudioEncoding.LINEAR16,
        "audio_channel_count": int(channels),
        "enable_word_time_offsets": True,
        "model": "video",
        "enable_automatic_punctuation":True
    }
    audio = {"uri": storage_uri}

    operation = client.long_running_recognize(config, audio)

    print(u"Waiting for operation to complete...")
    response = operation.result()
    return response

def subtitle_generation(response, bin_size=3):
    """We define a bin of time period to display the words in sync with audio. 
    Here, bin_size = 3 means each bin is of 3 secs. 
    All the words in the interval of 3 secs in result will be grouped togather."""
    transcriptions = []
    index = 0
 
    for result in response.results:
        try:
            if result.alternatives[0].words[0].start_time.seconds:
                # bin start -> for first word of result
                start_sec = result.alternatives[0].words[0].start_time.seconds 
                start_microsec = result.alternatives[0].words[0].start_time.nanos * 0.001
            else:
                # bin start -> For First word of response
                start_sec = 0
                start_microsec = 0 
            end_sec = start_sec + bin_size # bin end sec
            
            # for last word of result
            last_word_end_sec = result.alternatives[0].words[-1].end_time.seconds
            last_word_end_microsec = result.alternatives[0].words[-1].end_time.nanos * 0.001
            
            # bin transcript
            transcript = result.alternatives[0].words[0].word
            #transcript = transcript.tolower() 
            
            index += 1 # subtitle index

            for i in range(len(result.alternatives[0].words) - 1):
                try:
                    word = result.alternatives[0].words[i + 1].word
                    word_start_sec = result.alternatives[0].words[i + 1].start_time.seconds
                    word_start_microsec = result.alternatives[0].words[i + 1].start_time.nanos * 0.001 # 0.001 to convert nana -> micro
                    word_end_sec = result.alternatives[0].words[i + 1].end_time.seconds
                    word_end_microsec = result.alternatives[0].words[i + 1].end_time.nanos * 0.001

                    if word_end_sec < end_sec:
                        transcript = transcript + " " + word
                        #transcript = transcript.tolower() 

                    else:
                        previous_word_end_sec = result.alternatives[0].words[i].end_time.seconds
                        previous_word_end_microsec = result.alternatives[0].words[i].end_time.nanos * 0.001
                        
                        # append bin transcript
                        transcript = transcript.lower() 
                        transcriptions.append(srt.Subtitle(index, datetime.timedelta(0, start_sec, start_microsec), datetime.timedelta(0, previous_word_end_sec, previous_word_end_microsec), transcript))
                        # reset bin parameters
                        start_sec = word_start_sec
                        start_microsec = word_start_microsec
                        end_sec = start_sec + bin_size
                        transcript = result.alternatives[0].words[i + 1].word
                        
                        index += 1
                except IndexError:
                    pass
            # append transcript of last transcript in bin
            transcript = transcript.lower()
            transcriptions.append(srt.Subtitle(index, datetime.timedelta(0, start_sec, start_microsec), datetime.timedelta(0, last_word_end_sec, last_word_end_microsec), transcript))
            index += 1
        except IndexError:
            pass
    
    # turn transcription list into subtitles
    subtitles = srt.compose(transcriptions)
    return subtitles

#Paramters for the audio
chunk = 1024  

# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def create_audio_chunks(sound):
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")


########################################################################
########################################################################
#NLP CODE
WPLACEHOLDER = 'word_placeholder'

def _check_replace(w):
    return not bool(re.search(WPLACEHOLDER, w))

def _make_cleaning(s, c_dict):
    if _check_replace(s):
        s = s.translate(c_dict)
    return s

def _check_vocab(c_list, vocabulary, response='default'):
    try:
        words = set([w for line in c_list for w in line.split()])
        print('Total Words :',len(words))
        u_list = words.difference(set(vocabulary))
        k_list = words.difference(u_list)
    
        if response=='default':
            print('Unknown words:', len(u_list), '| Known words:', len(k_list))
        elif response=='unknown_list':
            return list(u_list)
        elif response=='known_list':
            return list(k_list)
    except:
        return []
    
def _make_dict_cleaning(s, w_dict):
    if _check_replace(s):
        s = w_dict.get(s, s)
    return s

def _print_dict(temp_dict, n_items=10):
    run = 0
    for k,v in temp_dict.items():
        print(k,'---',v)
        run +=1
        if run==n_items:
            break  
        
def to_lower(data):
    if verbose: print('#'*10 ,'Step - Lowering everything:')
    data = list(map(lambda x: x.lower(), data))
    return data

def clean_contractions(data):
    

    helper_contractions = {
     "aren't": 'are not',
     "Aren't": 'Are not',
     "AREN'T": 'ARE NOT',
     "C'est": "C'est",
     "C'mon": "C'mon",
     "c'mon": "c'mon",
     "can't": 'cannot',
     "Can't": 'Cannot',
     "CAN'T": 'CANNOT',
     "con't": 'continued',
     "cont'd": 'continued',
     "could've": 'could have',
     "couldn't": 'could not',
     "Couldn't": 'Could not',
     "didn't": 'did not',
     "Didn't": 'Did not',
     "DIDN'T": 'DID NOT',
     "don't": 'do not',
     "Don't": 'Do not',
     "DON'T": 'DO NOT',
     "doesn't": 'does not',
     "Doesn't": 'Does not',
     "else's": 'else',
     "gov's": 'government',
     "Gov's": 'government',
     "gov't": 'government',
     "Gov't": 'government',
     "govt's": 'government',
     "gov'ts": 'governments',
     "hadn't": 'had not',
     "hasn't": 'has not',
     "Hasn't": 'Has not',
     "haven't": 'have not',
     "Haven't": 'Have not',
     "he's": 'he is',
     "He's": 'He is',
     "he'll": 'he will',
     "He'll": 'He will',
     "he'd": 'he would',
     "He'd": 'He would',
     "Here's": 'Here is',
     "here's": 'here is',
     "I'm": 'I am',
     "i'm": 'i am',
     "I'M": 'I am',
     "I've": 'I have',
     "i've": 'i have',
     "I'll": 'I will',
     "i'll": 'i will',
     "I'd": 'I would',
     "i'd": 'i would',
     "ain't": 'is not',
     "isn't": 'is not',
     "Isn't": 'Is not',
     "ISN'T": 'IS NOT',
     "it's": 'it is',
     "It's": 'It is',
     "IT'S": 'IT IS',
     "I's": 'It is',
     "i's": 'it is',
     "it'll": 'it will',
     "It'll": 'It will',
     "it'd": 'it would',
     "It'd": 'It would',
     "Let's": "Let's",
     "let's": 'let us',
     "ma'am": 'madam',
     "Ma'am": "Madam",
     "she's": 'she is',
     "She's": 'She is',
     "she'll": 'she will',
     "She'll": 'She will',
     "she'd": 'she would',
     "She'd": 'She would',
     "shouldn't": 'should not',
     "that's": 'that is',
     "That's": 'That is',
     "THAT'S": 'THAT IS',
     "THAT's": 'THAT IS',
     "that'll": 'that will',
     "That'll": 'That will',
     "there's": 'there is',
     "There's": 'There is',
     "there'll": 'there will',
     "There'll": 'There will',
     "there'd": 'there would',
     "they're": 'they are',
     "They're": 'They are',
     "they've": 'they have',
     "They've": 'They Have',
     "they'll": 'they will',
     "They'll": 'They will',
     "they'd": 'they would',
     "They'd": 'They would',
     "wasn't": 'was not',
     "we're": 'we are',
     "We're": 'We are',
     "we've": 'we have',
     "We've": 'We have',
     "we'll": 'we will',
     "We'll": 'We will',
     "we'd": 'we would',
     "We'd": 'We would',
     "What'll": 'What will',
     "weren't": 'were not',
     "Weren't": 'Were not',
     "what's": 'what is',
     "What's": 'What is',
     "When's": 'When is',
     "Where's": 'Where is',
     "where's": 'where is',
     "Where'd": 'Where would',
     "who're": 'who are',
     "who've": 'who have',
     "who's": 'who is',
     "Who's": 'Who is',
     "who'll": 'who will',
     "who'd": 'Who would',
     "Who'd": 'Who would',
     "won't": 'will not',
     "Won't": 'will not',
     "WON'T": 'WILL NOT',
     "would've": 'would have',
     "wouldn't": 'would not',
     "Wouldn't": 'Would not',
     "would't": 'would not',
     "Would't": 'Would not',
     "y'all": 'you all',
     "Y'all": 'You all',
     "you're": 'you are',
     "You're": 'You are',
     "YOU'RE": 'YOU ARE',
     "you've": 'you have',
     "You've": 'You have',
     "y'know": 'you know',
     "Y'know": 'You know',
     "ya'll": 'you will',
     "you'll": 'you will',
     "You'll": 'You will',
     "you'd": 'you would',
     "You'd": 'You would',
     "Y'got": 'You got',
     'cause': 'because',
     "had'nt": 'had not',
     "Had'nt": 'Had not',
     "how'd": 'how did',
     "how'd'y": 'how do you',
     "how'll": 'how will',
     "how's": 'how is',
     "I'd've": 'I would have',
     "I'll've": 'I will have',
     "i'd've": 'i would have',
     "i'll've": 'i will have',
     "it'd've": 'it would have',
     "it'll've": 'it will have',
     "mayn't": 'may not',
     "might've": 'might have',
     "mightn't": 'might not',
     "mightn't've": 'might not have',
     "must've": 'must have',
     "mustn't": 'must not',
     "mustn't've": 'must not have',
     "needn't": 'need not',
     "needn't've": 'need not have',
     "o'clock": 'of the clock',
     "oughtn't": 'ought not',
     "oughtn't've": 'ought not have',
     "shan't": 'shall not',
     "sha'n't": 'shall not',
     "shan't've": 'shall not have',
     "she'd've": 'she would have',
     "she'll've": 'she will have',
     "should've": 'should have',
     "shouldn't've": 'should not have',
     "so've": 'so have',
     "so's": 'so as',
     "this's": 'this is',
     "that'd": 'that would',
     "that'd've": 'that would have',
     "there'd've": 'there would have',
     "they'd've": 'they would have',
     "they'll've": 'they will have',
     "to've": 'to have',
     "we'd've": 'we would have',
     "we'll've": 'we will have',
     "what'll": 'what will',
     "what'll've": 'what will have',
     "what're": 'what are',
     "what've": 'what have',
     "when's": 'when is',
     "when've": 'when have',
     "where'd": 'where did',
     "where've": 'where have',
     "who'll've": 'who will have',
     "why's": 'why is',
     "why've": 'why have',
     "will've": 'will have',
     "won't've": 'will not have',
     "wouldn't've": 'would not have',
     "y'all'd": 'you all would',
     "y'all'd've": 'you all would have',
     "y'all're": 'you all are',
     "y'all've": 'you all have',
     "you'd've": 'you would have',
     "you'll've": 'you will have'}
    if verbose: print('#' * 10, 'Step - Contractions:')
    # Apply spellchecker for contractions
    # Local (only unknown words)
    local_vocab = {}
    temp_vocab = _check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (_check_replace(k)) and ("'" in k)]
    temp_dict = {}
    for word in temp_vocab:
        if word in helper_contractions:
            temp_dict[word] = helper_contractions[word]
    data = list(map(lambda x: ' '.join([_make_dict_cleaning(i,temp_dict) for i in x.split()]), data))
    if verbose: _print_dict(temp_dict) 
    return data






