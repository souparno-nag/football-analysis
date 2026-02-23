import supervision as sv
import numpy as np

class MOTTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections):

        if len(detections) == 0:
            return []

        xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confidence = np.array([d["confidence"] for d in detections], dtype=np.float32)
        class_id = np.array([d["class"] for d in detections], dtype=np.int32)

        # 🔎 DEBUG HERE
        print("XYXY SHAPE:", xyxy.shape)
        print("CONF SHAPE:", confidence.shape)
        print("CLASS SHAPE:", class_id.shape)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        tracks = self.tracker.update_with_detections(sv_detections)

        return tracks