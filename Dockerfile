FROM ubuntu:18.04

MAINTAINER Shashi Raj
#RUN git clone https://github.com/shashi29/VideoProcessing.git 
#RUN cd checkout VideoProcessing && git checkout IntegV1

RUN apt-get update && apt-get install -y python3 python3-pip sudo

COPY VideoProcessing/ app/
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE 5000

CMD python3 app.py
