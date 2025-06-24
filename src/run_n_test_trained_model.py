from ultralytics import settings,YOLO

# settings.update({"wandb": False})

# Load a pretrained YOLO model
model = YOLO("runs/detect/train2/weights/best11n640100.pt")


# results = model(source="../test_footage/video/random_things.mp4", save=True, show=True) 


# for result in results:
#     # result.boxes.conf contains confidence scores for each detection
#     confidences = result.boxes.conf.cpu().numpy()
#     for conf in confidences:
#         print(f"Confidence: {conf * 100:.2f}%")



# Run inference on a video file and print confidence scores
for result in model(source="../test_footage/video/flera_dron.mp4",iou=0.3, stream=True, show=True):
    confidences = result.boxes.conf.cpu().numpy()
    for conf in confidences:
        print(f"Confidence: {conf:.2f}")