import numpy as np
import cv2
from scipy.spatial import Voronoi

class SpaceControl:
    def __init__(self, view_transformer):
        self.view_transformer = view_transformer
        self.court_width = 68
        self.court_length = 23.32

    def draw_voronoi(self, video_frames, tracks, mode='team'):
        """
        mode: 'team' (colors merged by team) or 'player' (distinct colors per player)
        """
        output_frames = []
        
        # Define transparency for overlay
        alpha = 0.4 

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            overlay = frame.copy()

            # 1. Collect points and team info
            points = []
            player_ids = []
            teams = []
            
            player_dict = tracks['players'][frame_num]
            
            for player_id, player in player_dict.items():
                if 'position_transformed' in player and player['position_transformed'] is not None:
                    # We need positions in 2D meter space for accurate calculation
                    points.append(player['position_transformed'])
                    player_ids.append(player_id)
                    teams.append(player.get('team', 0))

            if len(points) < 4: 
                # Voronoi needs at least 4 points to work reliably
                output_frames.append(frame)
                continue

            # 2. Add Dummy Points to close regions at the edge of the pitch
            # We add points way outside the pitch boundaries
            x_min, x_max = -10, self.court_length + 10
            y_min, y_max = -10, self.court_width + 10
            
            dummy_points = [
                [x_min, y_min], [x_max, y_min], 
                [x_min, y_max], [x_max, y_max],
                [x_min - 10, self.court_width/2], [x_max + 10, self.court_width/2]
            ]
            all_points = np.array(points + dummy_points)

            # 3. Compute Voronoi
            vor = Voronoi(all_points)

            # 4. Draw Regions
            # Iterate only through the real players (indices 0 to len(points))
            for i in range(len(points)):
                region_index = vor.point_region[i]
                region_vertices_indices = vor.regions[region_index]
                
                # If region is valid and doesn't contain -1 (infinite vertex)
                if -1 not in region_vertices_indices and len(region_vertices_indices) > 0:
                    # Get 2D vertices for this player's region
                    region_vertices_2d = [vor.vertices[v] for v in region_vertices_indices]
                    
                    # Convert 2D vertices back to Video Pixel Coordinates
                    region_vertices_px = self.view_transformer.inverse_transform(region_vertices_2d)

                    # Determine Color
                    if mode == 'team':
                        team_id = teams[i]
                        # Red for Team 1, Blue for Team 2 (BGR format)
                        color = (0, 0, 255) if team_id == 1 else (255, 0, 0)
                        if team_id == 0: color = (200, 200, 200) # Unknown
                    else:
                        # Random color for individual player
                        np.random.seed(player_ids[i])
                        color = np.random.randint(0, 255, 3).tolist()

                    # Draw Polygon on Overlay
                    cv2.fillPoly(overlay, [region_vertices_px], color)

            # 5. Blend Overlay with Original Frame
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            output_frames.append(frame)

        return output_frames