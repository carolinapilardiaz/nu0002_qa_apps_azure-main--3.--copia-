FROM python:3.10

RUN apt-get update --fix-missing
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python

EXPOSE 8000

WORKDIR /app
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]