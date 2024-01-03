FROM python:3.10

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

RUN apt-get update --fix-missing
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql18

COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python

EXPOSE 8000

WORKDIR /app
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]