import numpy as np

class SpeedEstimator:
    def __init__(self, fps):
        self.fps = fps
        self.history = {}

    def update(self, player_id, position):
        if player_id not in self.history:
            self.history[player_id] = []
        
        self.history[player_id].append(position)

    def compute_speed(self, player_id):
        positions = self.history[player_id]

        if len(positions) < 2:
            return 0

        p1 = np.array(positions[-2])
        p2 = np.array(positions[-1])

        distance = np.linalg.norm(p2 - p1)
        speed = distance * self.fps

        return speed