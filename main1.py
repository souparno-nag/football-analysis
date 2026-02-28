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

    # Save Video
    output_video_path = "./data/output/output1.mp4"
    save_video(video_frames, output_video_path)

if __name__ == "__main__":
    main()