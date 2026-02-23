import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class HeatmapGenerator:
    def __init__(self, pitch_width=105, pitch_height=68):
        self.grid = np.zeros((pitch_height, pitch_width))

    def update(self, position):
        x, y = int(position[0]), int(position[1])
        if 0 <= x < 105 and 0 <= y < 68:
            self.grid[y][x] += 1

    def generate(self, save_path):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.grid, cmap="Reds")
        plt.savefig(save_path)
        plt.close()