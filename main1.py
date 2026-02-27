from src1.utils import read_video, save_video

def main():
    # Read Video
    input_video_path = "./data/raw_video/match1.mp4"
    video_frames = read_video(input_video_path)
    # Save Video
    output_video_path = "./data/output/output1.mp4"
    save_video(video_frames, output_video_path)

if __name__ == "__main__":
    main()