# Statistical Analysis of a Football Game from Video

This project uses object detection models (YOLO) and computer vision (OpenCV) to perform statistical analysis of a given football game.

## Procedure

### Model Training (YOLOV8)

This module fine tuned a pre-trained neural network to specifically detect football related objects using a Roboflow dataset.

1. **Environment Setup**: Installs `ultralytics` (for YOLO) and `roboflow` (for data), and loads API keys.
2. **Data Acquisition**: Downloads the `football-players-detection-3zvbc` dataset from Roboflow in YOLOv8 format.
3. **Data Preparation**: Reorganizes the downloaded dataset into a standard `train/test/valid` folder structure expected by the training script.
4. **Configuration**: Sets up a training run using the pre-trained `yolov8m.pt` (Medium) model as a starting point.
5. **Training Execution**: Runs the training loop for **100 epochs** with an image size of **640px**. The model learns to minimize box, class, and focal loss.
6. **Model Export**: Saves the best-performing weights (`best.pt`) and copies them to the project's `models/` folder for use in the tracking module.

### Tracking (Players, Ball and Referee)

This module detects objects in each frame and assigns them a consistent ID over time.

1. **Detection:** The system utilizes a YOLO model (loaded from `best.pt`) to detect objects in every frame. It identifies classes: `player`, `referee`, and `ball`.
2. **Class Override:** Detected `goalkeeper` objects are programmatically re-classified as `player` to simplify tracking logic.
3. **Tracking:** The detections are passed to **ByteTrack** (via the `supervision` library). ByteTrack associates detections across consecutive frames to assign unique IDs.
4. **Position Extraction:**
    - **Players/Referees:** The position is defined as the center of the bottom edge of the bounding box (representing the feet).
    - **Ball:** The position is defined as the center of the bounding box.
5. **Interpolation:** Missing ball detections are filled in using linear interpolation (`pandas.interpolate`) to ensure smooth tracking even when the ball is momentarily occluded or not detected.

### Colour Detection and Team Assigning

This module differentiates teams based on jersey color using K-Means clustering.

1. **Player Cropping:** For a specific frame, the system crops the image to the bounding box of each player.
2. **Shirt Isolation:** It isolates the top half of the cropped image (assuming the shirt is in the top half) to avoid including shorts or socks.
3. **Individual Color Clustering:** K-Means clustering (K=2) is applied to the shirt pixels to separate the "shirt color" from the "background/non-shirt color". The cluster furthest from the image corners is selected as the player's color.
4. **Team Clustering:** Once colors are extracted for all players, a second K-Means model (K=2) clusters all these individual colors into two main groups (Team 1 and Team 2).
5. **Prediction:** For every subsequent frame, the system extracts a player's color and uses the trained K-Means model to predict which team cluster it belongs to.

### Assigning Ball to Player

This logic determines if a player is in possession of the ball.

1. **Thresholding:** A maximum pixel distance threshold (e.g., 70 pixels) is defined.
2. **Distance Calculation:** For every frame, the system calculates the distance between the ball's position and the feet of every player.
3. **Assignment:**
    - It identifies the player closest to the ball.
    - If that minimum distance is within the threshold, the `has_ball` attribute is set to `True` for that player ID.
    - If the distance exceeds the threshold, no player is assigned possession.

### Camera Movement Estimation

This module compensates for camera panning/tilting to calculate real-world movements.

1. **Feature Detection:** In the first frame, "good features to track" (corners/edges) are identified using `cv2.goodFeaturesToTrack`. Masking is used to ignore features on players or the ball, focusing only on the static background (pitch).
2. **Optical Flow:** Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`) tracks these features from the previous frame to the current frame.
3. **Movement Calculation:** The median or average displacement (x, y) of these features is calculated. This represents the camera's movement.
4. **Adjustment:** This movement vector is subtracted from the raw positions of players/ball to get their "adjusted" positions relative to a static camera view.

### View Transformation

This maps 2D video pixel coordinates to real-world meters (birds-eye view).

1. **Perspective Matrix:** A transformation matrix is computed using `cv2.getPerspectiveTransform`. This requires:
    - **Source Points:** Four pixel coordinates manually selected from the video (e.g., corners of the penalty box).
    - **Target Points:** The corresponding real-world coordinates in meters (e.g., `(0,0)`, `(0, 68)`, etc.).
2.**Transformation:** For every tracked object, its adjusted pixel position is multiplied by this matrix (`cv2.perspectiveTransform`) to output a position in meters `(x, y)`.

### Speed and Distance Estimation

This calculates physical metrics using the transformed coordinates.

1. **Windowing:** To reduce noise, speed is calculated over a small window of frames (e.g., every 5 frames) rather than frame-by-frame.
2. **Distance Calculation:** Euclidean distance is measured between a player's real-world position at `frame_i` and `frame_i+5`.
3. **Speed Formula:**
    - `Speed (m/s) = Distance (m) / Time (s)` (where time is derived from the video framerate, e.g., 24fps).
    - Converted to `km/h` by multiplying by 3.6.
4. **Accumulation:** The distance covered in each window is added to a running total for that player.

### Pass Detection

This identifies passing events based on possession changes.

1. **Possession History:** The system iterates through the frames to build a timeline of "who has the ball".
2. **Transition Logic:** It looks for a specific sequence:
    - **State A:** Player X has the ball.
    - **State B:** (Optional) No one has the ball (ball in transit).
    - **State C:** Player Y has the ball.
3. **Validation:** If `Player X != Player Y`, a pass is recorded with a start frame (when X lost it) and end frame (when Y caught it).

### Tactical Heatmap

This visualizes player density on the field over time.

1. **Grid Initialization:** A 2D array (image) is created representing the pitch dimensions in meters, scaled up for resolution.
2. **Mapping:** For every frame, every player's real-world position `(x, y)` is mapped to a pixel on this grid.
3. **Gaussian Blur:** Instead of single dots, a Gaussian "blob" is added to the grid at each position to simulate heat/density.
4. **Blending:** The accumulated heatmap is color-coded (e.g., using a JET colormap) and overlaid onto a green pitch template or the original video.

### Space Control

This divides the pitch into regions based on which player is closest.

1. **Point Collection:** All players' transformed positions (meters) are collected for the current frame.
2. **Dummy Points:** Artificial points are added far outside the pitch boundaries. This forces the mathematical regions of edge players to close into finite shapes rather than extending to infinity.
3. **Tessellation:** `scipy.spatial.Voronoi` calculates the polygonal regions for each point.
4. **Inverse Transformation:** The vertices of these regions (which are in meters) are transformed _back_ to video pixel coordinates using the inverse of the matrix from Step 5.
5. **Drawing:** These polygons are drawn on the video frame, colored semi-transparently according to the team of the player owning that region.
