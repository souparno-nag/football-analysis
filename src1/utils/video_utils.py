import cv2

# takes video path as input and returns a list of frames
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

# takes a list of frames and saves it as a video at the specified path
def save_video(output_video_frames, output_video_path, fps=25):
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()