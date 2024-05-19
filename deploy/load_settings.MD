docker build -t yolo .

docker run -it --gpus all --restart=unless-stopped --name yolo -p 8090:8090 yolo