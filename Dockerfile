FROM python:3.10.16-slim

RUN apt-get update && \
    apt-get install -y build-essential ffmpeg git libsox-dev sox && \
    apt-get clean

WORKDIR /app

RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

WORKDIR /app/CosyVoice

RUN git submodule update --init --recursive && \
    pip install Flask==3.1.0 celery==5.4.0 pynini==2.1.5 && \
    pip install -r requirements.txt && \
    mkdir pretrained_models

RUN pip install boto3

COPY app.py .

CMD [ "python", "app.py" ]
