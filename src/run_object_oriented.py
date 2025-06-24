from ultralytics import settings,YOLO

class yolo_model:
    
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def train(self, data_path, epochs=5, imgsz=320):
        self.data_path = data_path
        results = self.model.train(data=self.data_path, epochs=epochs, imgsz=imgsz)
        return results
    
    def run_inference(self, source, stream=False, show=True):
        results = self.model(source=source, stream=stream, show=show)
        return results

    def print_confidences(self, results):
        for result in results:
            confidences = result.boxes.conf.cpu().numpy()
            for conf in confidences:
                print(f"Confidence: {conf:.2f}")
    
    def run_conf_n_inference(self, source, stream=False, show=True):
        for result in self.model(source=source, stream=stream, show=show):
            confidences = result.boxes.conf.cpu().numpy()   
            for conf in confidences:
                print(f"Confidence: {conf:.2f}")

    def get_detections(self, source, conf_threshold=0.0):
        results = self.model(source=source, stream=False)
        all_boxes = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = boxes.conf[i].item()
                if conf >= conf_threshold:
                    all_boxes.append([*xyxy, conf])
        return all_boxes
