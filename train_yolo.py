from ultralytics import YOLO


model = YOLO("yolov8m.pt")

model.train(data="/workspace/data/data.yaml", epochs=5, imgsz=640)