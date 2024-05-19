# server

from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import base64
from Image_analysis import *
import torch
import gc

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello", "Diplom!"}


@app.post("/YOLOv8m_inferences")
async def process(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_dimensions = str(img.shape)
    image_with_bbox, predcited_labels = YOLOv8m_analise(img)

    _, image_with_bbox = cv2.imencode('.JPEG', image_with_bbox)
    image_with_bbox = base64.b64encode(image_with_bbox)

    gc.collect()
    torch.cuda.empty_cache()
    return {"image_with_bbox":image_with_bbox,
            'image_dimensions': img_dimensions,
            "predcited_labels": predcited_labels}


@app.post("/YOLOv8m_cls_inferences")
async def process(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_dimensions = str(img.shape)
    image_with_labels, predcited_labels = YOLOv8m_cls_analise(img)

    _, image_with_labels = cv2.imencode('.JPEG', image_with_labels)
    image_with_labels = base64.b64encode(image_with_labels)

    gc.collect()
    torch.cuda.empty_cache()
    return {"image_with_labels":image_with_labels,
            'image_dimensions': img_dimensions,
            "predcited_labels": predcited_labels}