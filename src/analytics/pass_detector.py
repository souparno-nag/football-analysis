import numpy as np

class PassDetector:
    def __init__(self, threshold=3):
        self.last_possession = None
        self.threshold = threshold

    def detect(self, ball_pos, players):
        for player_id, pos in players.items():
            if np.linalg.norm(np.array(ball_pos) - np.array(pos)) < self.threshold:
                if self.last_possession is None:
                    self.last_possession = player_id
                elif self.last_possession != player_id:
                    print(f"Pass from {self.last_possession} to {player_id}")
                    self.last_possession = player_id