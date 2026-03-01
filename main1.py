import cv2
import numpy as np
from src1.utils import read_video, save_video
from src1.tracker import Tracker
from src1.team_assigner import TeamAssigner
from src1.player_ball_assigner import PlayerBallAssigner

def main():
    # Read Video
    input_video_path = "./data/raw_video/match1.mp4"
    video_frames = read_video(input_video_path)    
    
    model_path = "./src1/models/best.pt"
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stubs=True, 
                                        stub_path="./data/stubs/tracks_stubs.pkl")
    
    # Interpolate Ball Positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # # Save Cropped Image of a Player
    # for track_id, player in tracks["players"][0].items():
    #     bbox = player["bbox"]
    #     frame = video_frames[0]
    #     # Crop bbox from frame
    #     croped_player = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    #     # Save cropped image
    #     save_path = "./data/image/cropped_img.jpg"
    #     cv2.imwrite(save_path, croped_player)
    #     break

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    
    # Draw Output
    # Draw Object Tracks on Frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save Video
    output_video_path = "./data/output/output5.mp4"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()