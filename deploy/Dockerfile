FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY . .
CMD python main.py