import cv2
from src1.utils import read_video, save_video
from src1.tracker import Tracker

def main():
    # Read Video
    input_video_path = "./data/raw_video/match1.mp4"
    video_frames = read_video(input_video_path)    
    
    model_path = "./src1/models/best.pt"
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stubs=True, 
                                        stub_path="./data/stubs/tracks_stubs.pkl")
    
    # Save Cropped Image of a Player
    for track_id, player in tracks["players"][0].items():
        bbox = player["bbox"]
        frame = video_frames[0]
        # Crop bbox from frame
        croped_player = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # Save cropped image
        save_path = "./data/image/cropped_img.jpg"
        cv2.imwrite(save_path, croped_player)
        break
    
    # Draw Output
    # Draw Object Tracks on Frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    output_video_path = "./data/output/output2.mp4"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()