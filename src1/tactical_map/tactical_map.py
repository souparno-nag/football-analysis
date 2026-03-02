import cv2
import numpy as np

class TacticalMap:
    def __init__(self):
        # Match these to your ViewTransformer dimensions
        self.court_width = 68
        self.court_length = 23.32 
        
        # Scale for the output image (pixels per meter)
        self.scale = 10 
        self.map_w = int(self.court_width * self.scale)
        self.map_h = int(self.court_length * self.scale)

    def draw_pitch(self, image):
        # Draw outlines
        cv2.rectangle(image, (0,0), (self.map_w, self.map_h), (255,255,255), 2)
        # Draw center line (approximate based on your view transformer logic)
        # Since 23.32m is likely a partial field, we just draw a grid
        for i in range(0, self.map_w, 100):
            cv2.line(image, (i, 0), (i, self.map_h), (255,255,255, 50), 1)
        return image

    def draw_tactical_map(self, tracks, output_path=None):
        """
        Generates a heatmap of player positions.
        """
        # Create a black canvas
        heatmap_canvas = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)

        # Separate positions by team
        team_positions = {1: [], 2: []}

        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track_info in player_track.items():
                
                # Ensure we have a team and a transformed position
                if 'team' not in track_info or 'position_transformed' not in track_info:
                    continue
                
                pos = track_info['position_transformed']
                if pos is None: 
                    continue
                
                team = track_info['team']
                
                # Convert meters to map pixels
                # Note: ViewTransformer target vertices: 
                # [0, width], [0,0], [length, 0], [length, width]
                # We need to map coordinate system correctly.
                # Assuming pos[0] is x (length) and pos[1] is y (width) based on standard geometry
                
                map_x = int(pos[0] * self.scale)
                map_y = int(pos[1] * self.scale)

                # Clamp values to image size
                map_x = max(0, min(map_x, self.map_w - 1))
                map_y = max(0, min(map_y, self.map_h - 1))

                if team in team_positions:
                    team_positions[team].append((map_x, map_y))

        # --- Draw Heatmaps ---
        # We will create two overlays, one for each team
        
        team_colors = {1: (0, 0, 255), 2: (255, 0, 0)} # Red and Blue

        final_map = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
        final_map[:] = (0, 100, 0) # Green background pitch

        final_map = self.draw_pitch(final_map)

        for team_id, points in team_positions.items():
            if not points: continue
            
            # Create a layer for this team
            overlay = np.zeros((self.map_h, self.map_w), dtype=np.float32)
            
            # Add points
            for (x, y) in points:
                # Add a gaussian blob at this location
                # We use a simple circle add for speed, creating density
                cv2.circle(overlay, (x, y), 15, 1, -1) # type: ignore
            
            # Blur to create heat effect
            overlay = cv2.GaussianBlur(overlay, (45, 45), 0)
            
            # Normalize
            overlay = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
            overlay = np.uint8(overlay)

            # Apply color map
            colored_heatmap = cv2.applyColorMap(overlay, cv2.COLORMAP_JET) # type: ignore
            
            # Mask: Only show heatmap where there is density
            mask = overlay > 10 
            
            # Blend
            alpha = 0.5
            # This is a simple blend. For better visualization, you might want separate maps
            # But here we blend onto the green pitch
            for c in range(3):
                final_map[:,:,c] = np.where(mask, 
                                            (alpha * colored_heatmap[:,:,c] + (1-alpha) * final_map[:,:,c]), 
                                            final_map[:,:,c])

        if output_path:
            cv2.imwrite(output_path, final_map)
        
        return final_map