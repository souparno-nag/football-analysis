import cv2
import numpy as np
from src1.utils import read_video, save_video
from src1.tracker import Tracker
from src1.team_assigner import TeamAssigner
from src1.player_ball_assigner import PlayerBallAssigner
from src1.camera_movement_estimator import CameraMovementEstimator
from src1.view_transformer import ViewTransformer
from src1.speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    # Read Video
    input_video_path = "./data/raw_video/match1.mp4"
    video_frames = read_video(input_video_path)    
    
    model_path = "./src1/models/best.pt"
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stubs=True, 
                                        stub_path="./data/stubs/tracks_stubs.pkl")
    # Get object positions 
    tracker.add_position_to_tracks(tracks)
    

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='./data/stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate Ball Positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
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
    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save Video
    output_video_path = "./data/output/output8.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()