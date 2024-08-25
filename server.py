from typing import Annotated
import torch
import uvicorn
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File
from model.autoencoder import AutoEncoder
from CORS import get
from PIL import Image

face_model = AutoEncoder()
face_model.load_state_dict(torch.load("face.pt"))

detect_model = YOLO(model="detect.pt", task="detect")
face_detect_model = YOLO(model="yolov8m.pt", task="detect")

app = FastAPI()

@app.post("/post")
async def passport(file1: Annotated[bytes, File()], file2: Annotated[bytes, File()]):
    detect_results = detect_model.predict(source=Image.frombytes("RGB", (640, 640), file1))
    text = ""
    for result in detect_results:
        text += get(Image.frombytes("RGB", (640, 640), file1).crop(result.boxes))
    detect_result = ""
    detect_results = face_detect_model.predict(Image.frombytes("RGB", (128, 128), file2))
    for result in detect_results:
        if result.boxes.cls == "person":
            detect_result = Image.frombytes("RGB", (640, 640),file1).crop(result.boxes)
    
    return text, torch.nn.functional.cosine_similarity(face_model(file2), face_model(detect_result))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
