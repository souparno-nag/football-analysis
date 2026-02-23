import cv2
from src.detection.player_detector import PlayerBallDetector
from src.tracking.mot_tracker import MOTTracker

video = cv2.VideoCapture("data/raw_video/match.mp4")

detector = PlayerBallDetector("yolov8n.pt")
tracker = MOTTracker()

while True:
    ret, frame = video.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections)

    if tracks is not None and len(tracks) > 0:
        for i in range(len(tracks)):
            x1, y1, x2, y2 = map(int, tracks.xyxy[i])
            track_id = tracks.tracker_id[i]

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, str(track_id), (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Match Analysis", frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()