FROM ubuntu:18.04

MAINTAINER Shashi Raj
#RUN git clone https://github.com/shashi29/VideoProcessing.git 
#RUN cd checkout VideoProcessing && git checkout IntegV1
RUN apt update
RUN apt install -y ffmpeg
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
COPY VideoProcessing/ app/
WORKDIR /app
RUN ls
RUN pip3 install -r requirements.txt
EXPOSE 5000

CMD python3 app.py
