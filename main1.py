from ultralytics import YOLO
import os

model = YOLO('yolov8m.pt').to('cuda')
input_path = "./data/raw_video/match1.mp4"

result = model.predict(
    input_path,
    save=True,
    project=os.path.abspath('data'),  # absolute path
    name='output',
    exist_ok=True
)
print(result[0])
print("=================================================")
for box in result[0].boxes: # type: ignore
    print(box)