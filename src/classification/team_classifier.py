import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamClassifier:
    def __init__(self):
        self.team_colors = None

    def extract_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(pixels)

        dominant = kmeans.cluster_centers_[0]
        return dominant

    def assign_teams(self, color_list):
        kmeans = KMeans(n_clusters=2, n_init=10)
        labels = kmeans.fit_predict(color_list)
        return labels