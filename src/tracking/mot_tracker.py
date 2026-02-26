from deep_sort_realtime.deepsort_tracker import DeepSort

class MOTTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=60,                # survive occlusion
            n_init=5,                  # confirm track
            max_cosine_distance=0.15,  # strong appearance match
            nn_budget=300,
            embedder="mobilenet",
            half=True                  # FP16 on RTX
        )

    def update(self, detections, frame):

        player_detections = []

        for d in detections:
            if d["class"] != 0:  # track only players
                continue

            x1, y1, x2, y2 = d["bbox"]
            w = x2 - x1
            h = y2 - y1

            player_detections.append((
                [x1, y1, w, h],
                d["confidence"],
                "player"
            ))

        tracks = self.tracker.update_tracks(
            player_detections,
            frame=frame
        )

        return tracks