FROM python:3.10

ENV DISPLAY=:1

RUN apt-get update --fix-missing
RUN apt-get install ffmpeg xvfb libsm6 libxext6 -y

COPY ./requirements.txt requirements.txt
RUN pip3 install -U -r requirements.txt --default-timeout=100

RUN jt -t monokai -f fira -fs 10 -nf ptsans -nfs 11 -N -kl -cursw 2 -cursc r -cellw 95% -T

EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/workdir --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]