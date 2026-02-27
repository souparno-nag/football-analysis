from ultralytics import YOLO
import os

model_path="./src1/models/best.pt"
model = YOLO(model=model_path).to('cuda')
input_path = "./data/raw_video/match1_tiny.mp4"

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