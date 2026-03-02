import numpy as np
import cv2
from src1.utils import get_center_of_bbox

class PassDetector:
    def __init__(self):
        self.passes = [] # List of tuples: (frame_num, passer_id, receiver_id)

    def detect_passes(self, tracks):
        """
        Logic:
        1. Identify who has the ball in every frame.
        2. If possession changes from Player A -> None -> Player B, 
           and Player A != Player B, it is a pass.
        """
        player_tracks = tracks['players']
        ball_possessor_history = []

        # 1. Build a simplified list of who has the ball per frame
        for frame_num in range(len(player_tracks)):
            possessor_id = -1
            # Check which player has the ball in this frame
            for player_id, player_data in player_tracks[frame_num].items():
                if player_data.get('has_ball', False):
                    possessor_id = player_id
                    break
            ball_possessor_history.append(possessor_id)

        # 2. Analyze transitions
        last_possessor = -1
        pass_start_frame = -1

        for frame_num, current_possessor in enumerate(ball_possessor_history):
            
            # Case 1: Someone has the ball
            if current_possessor != -1:
                # If it's a new player holding the ball (and not the start of video)
                if last_possessor != -1 and last_possessor != current_possessor:
                    # Valid Pass Detected from last_possessor to current_possessor
                    # We define the pass frame as the moment the receiver gets it
                    self.passes.append({
                        "start_frame": pass_start_frame,
                        "end_frame": frame_num,
                        "passer_id": last_possessor,
                        "receiver_id": current_possessor
                    })
                
                # Update state
                last_possessor = current_possessor
                pass_start_frame = frame_num # Reset start frame for next potential pass
            
            # Case 2: No one has the ball (ball in transit)
            # We do nothing, we just wait for the ball to land at a new player
            else:
                continue

        return self.passes

    def draw_passes(self, video_frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            # Check if this frame is part of a completed pass animation window
            # We will draw the pass for 30 frames after it happens to make it visible
            for pass_info in self.passes:
                if pass_info['end_frame'] <= frame_num < pass_info['end_frame'] + 30:
                    
                    passer_id = pass_info['passer_id']
                    receiver_id = pass_info['receiver_id']

                    # Get positions (Use the frame where the pass ENDED to get receiver pos, 
                    # and frame where pass STARTED to get passer pos)
                    
                    # Guard clause in case track was lost
                    if (passer_id in tracks['players'][pass_info['start_frame']] and 
                        receiver_id in tracks['players'][pass_info['end_frame']]):
                        
                        passer_bbox = tracks['players'][pass_info['start_frame']][passer_id]['bbox']
                        receiver_bbox = tracks['players'][pass_info['end_frame']][receiver_id]['bbox']

                        p1 = get_center_of_bbox(passer_bbox)
                        p2 = get_center_of_bbox(receiver_bbox)

                        # Draw Line
                        cv2.line(frame, p1, p2, (0, 255, 255), 3)
                        
                        # Draw Text
                        mid_point = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                        cv2.putText(frame, "Pass!", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            output_frames.append(frame)
        return output_frames