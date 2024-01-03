FROM nvidia/cuda:11.7.1-base-ubuntu20.04

ENV PYTHON_VERSION=3.10

RUN DEBIAN_FRONTEND=noninteractive 
ENV TZ=America/Bogota
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update --fix-missing
RUN apt-get install -y --no-install-recommends tzdata keyboard-configuration
RUN apt-get install python${PYTHON_VERSION} python3-pip -y
RUN apt-get install ffmpeg xvfb libsm6 libxext6 -y
RUN apt-get install git -y

RUN pip3 install setuptools-rust
RUN pip3 install git+https://github.com/openai/whisper.git 

COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --default-timeout=100

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python

EXPOSE 8000

WORKDIR /app
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
