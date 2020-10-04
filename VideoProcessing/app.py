import os
import requests
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.tools import cvsecs
from moviepy.video.VideoClip import TextClip, VideoClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from run import *
from utility import *

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__, static_url_path="/static")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 20mb
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        try:
            url = request.form['url']
            r = requests.get(url)
            print(r.text)
            maskText = r.text
            maskText = maskText.split(" ")
            with open('mask_word.txt', 'w') as writer:
                for word in maskText:
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

    #os.system("rm -rf *.wav")
    #os.system("rm -rf *.mp4")
    return render_template('index.html')


def process_file(path, filename):
    process_video(path, filename)

def add_srt_video(srt_path, video_path):
    print("[INFO] Adding srt file to the final video")
    result_video_name = video_path#video_path[:-4] + '_result.mp4'
    generator = lambda txt: TextClip(txt, font = 'Arial', fontsize = 16, color = 'white')
    subtitles = SubtitlesClip(srt_path, generator)
    video = VideoFileClip(video_path)
    result = CompositeVideoClip([video, subtitles])
    result.write_videofile(video_path, fps = video.fps)


def process_video(video_path, filename):
   
    video_name = os.path.basename(video_path)
    video_name = video_name.split(".")[0]
    raw_audio_name = f'{video_name}_audio.wav'
    beep_path = "beep.wav"
    raw_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_audio_name)
    processed_audio_name = f'{video_name}_final.wav'
    processed_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_audio_name)
    BUCKET_NAME = "audio_2020"
    no_audio_video_path = video_path[:-4] + '_No_Audio.mp4'
    processed_video = os.path.join(app.config['DOWNLOAD_FOLDER'],filename)
    
    channels, bit_rate, sample_rate = video_info(video_path)
    blob_name = video_to_audio(video_path, raw_audio_path, channels, bit_rate, sample_rate)
    
    gcs_uri = f"gs://{BUCKET_NAME}/{raw_audio_name}"
    response = long_running_recognize(gcs_uri, channels, sample_rate)
    response_df = word_timestamp(response)

    #Add srt to the video
    srt = subtitle_generation(response)
    with open("subtitles.srt", "w") as f:
        f.write(srt)
    

    #mask audio
    mask_audio = process_audio(raw_audio_path, beep_path, response_df)
    mask_audio.export(processed_audio_path, format="wav")
    #Remove audio 
    command = f"ffmpeg -i {video_path} -vcodec copy -an -y {no_audio_video_path}"
    os.system(command)
    command = f"ffmpeg -i {no_audio_video_path} -i {processed_audio_path} -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k -y {processed_video}"
    os.system(command)
    add_srt_video("subtitles.srt", processed_video)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
