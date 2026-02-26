from ultralytics import YOLO

class PlayerBallDetector:
    def __init__(self, model_path="yolov8m.pt"):
        self.model = YOLO(model_path)
        self.model.to("cuda")

    def detect(self, frame):
        results = self.model(
            frame,
            device="cuda",
            imgsz=1280,      # Better for 1080p players
            conf=0.5,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])

            # Keep only players and ball
            if cls not in [0, 32]:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class": cls
            })

        return detections