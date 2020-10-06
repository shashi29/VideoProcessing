import os
import requests
import zlib
import zipfile
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from run import *
from utility import *

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
ALLOWED_EXTENSIONS = {'mp4', 'srt'}

app = Flask(__name__, static_url_path="/static")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 20mb
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compress(file_names_list, zip_file_name):
    compression = zipfile.ZIP_DEFLATED
    zf = zipfile.ZipFile(zip_file_name, mode="w")
    try:
        for file_name in file_names_list:
            zf.write(file_name, file_name, compress_type=compression)
    
    except FileNotFoundError:
        print("An error occurred")
    finally:
        zf.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        try:
            maskText = request.form['text-box']
            print(f"[INFO] {maskText}") #.text)
            maskText = maskText.split(" ")
            with open('mask_word.txt', 'w') as writer:
                print("[INFO] writing mask word")
                for word in maskText:
                    print(f"[INFO] {word}")
                    word = word + '\n'
                    writer.write(word)
            
            writer.close()
            
        except Exception as ex:
            print(ex)
            
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)
            return redirect(url_for('uploaded_file', filename=filename))

def process_file(path, filename):
    process_video(path, filename)
    
def process_video(video_path, filename):
   
    #Delete all mp4 files and audio
    video_name = os.path.basename(video_path)
    video_name = video_name.split(".")[0]
    raw_audio_name = f'{video_name}_audio.wav'
    beep_path = "beep.wav"
    raw_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_audio_name)
    processed_audio_name = f'{video_name}_final.wav'
    processed_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_audio_name)
    BUCKET_NAME = "audio_2020"
    no_audio_video_path = video_path[:-4] + '_No_Audio.mp4'
    final_video_name = video_name + "_final_result.mp4"
    processed_video = os.path.join(app.config['DOWNLOAD_FOLDER'], final_video_name)
    
    print("[INFO] Extracting audio output from video")
    channels, bit_rate, sample_rate = video_info(video_path)
    print("[INFO] Uploading audio file to the cloud")
    blob_name = video_to_audio(video_path, raw_audio_path, channels, bit_rate, sample_rate)
    
    print("[INFO] Running google speech to text API")
    gcs_uri = f"gs://{BUCKET_NAME}/{raw_audio_name}"
    response = long_running_recognize(gcs_uri, channels, sample_rate)
    response_df = word_timestamp(response)
    
    #mask audio
    mask_audio = process_audio(raw_audio_path, beep_path, response_df)
    mask_audio.export(processed_audio_path, format="wav")
    #Remove audio 
    command = f"ffmpeg -i {video_path} -vcodec copy -an -y {no_audio_video_path}"
    os.system(command)
    command = f"ffmpeg -i {no_audio_video_path} -i {processed_audio_path} -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k -y {processed_video}"
    os.system(command)
    print("[INFO] Final video is ready to download")
    
    #Add subtitle writer
    srt = subtitle_generation(response)
    srt_file_name = os.path.join(app.config['DOWNLOAD_FOLDER'], "subtitles.srt")
    with open(srt_file_name, "w") as f:
        f.write(srt)

    #Create zip file for video and srt file
    file_to_combine_list = [final_video_name, "subtitles.srt"]
    zip_file_name = filename[:-4] + ".zip"
    compress(file_to_combine_list, zip_file_name)
    
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    filename = filename[:-4] + ".zip"
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
