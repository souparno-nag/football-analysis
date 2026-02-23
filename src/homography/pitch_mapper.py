import cv2
import numpy as np

class PitchMapper:
    def __init__(self, src_points, dst_points):
        self.H, _ = cv2.findHomography(src_points, dst_points)

    def map_point(self, point):
        pt = np.array([[point]], dtype="float32")
        mapped = cv2.perspectiveTransform(pt, self.H)
        return mapped[0][0]