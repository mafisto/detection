FROM python:3.7
WORKDIR /project
COPY /project/requirements.txt requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements.txt
COPY /project .
ADD https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights /project/data
ENTRYPOINT ["python"]
CMD ["main.py"]