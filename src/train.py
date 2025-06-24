from ultralytics import settings,YOLO

# settings.update({"wandb": False})

# Load a YOLO model
model = YOLO("yolov5l.pt")


results = model.train(data="../data.yaml", epochs=1, imgsz=320)