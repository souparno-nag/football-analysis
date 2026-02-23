import pandas as pd

class MatchStatistics:
    def __init__(self):
        self.passes = []
        self.shots = []
        self.distance = {}

    def register_pass(self, p_from, p_to):
        self.passes.append((p_from, p_to))

    def add_distance(self, player_id, dist):
        if player_id not in self.distance:
            self.distance[player_id] = 0
        self.distance[player_id] += dist

    def export(self, path="outputs/statistics.csv"):
        df = pd.DataFrame({
            "Total Passes": [len(self.passes)]
        })
        df.to_csv(path, index=False)