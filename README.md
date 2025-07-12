# Cross-Camera Player Mapping

Mapping players across broadcast and tacticam views using YOLOv8 detections and feature matching.

---

### Objective

Given two clips (`broadcast.mp4` and `tacticam.mp4`) of the same gameplay from different camera angles, the goal is to:

- Detect players in both videos using a YOLO-based object detection model.
- Extract visual features for each detected player crop.
- Match each player from the tacticam video to their corresponding player in the broadcast video, assigning consistent player IDs across both feeds.

---

### 🛠️ Project Structure
cross-camera-player-mapping/
│
├── detections_broadcast/ # Cropped player images from broadcast.mp4
├── detections_tacticam/ # Cropped player images from tacticam.mp4
│
├── features_broadcast.pkl # Extracted features for broadcast crops
├── features_tacticam.pkl # Extracted features for tacticam crops
│
├── matched_players.pkl # Saved matching results
├── matched_players.csv # CSV export of matching results
│
├── scripts/
│ ├── detect_players.py # Runs YOLOv8 detection on videos
│ ├── extract_features.py # Extracts embeddings from crops
│ ├── match_players.py # Computes similarity and matches players
│ ├── visualize_matches.py # Plots visual examples of matched players
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── ...  


---

## ⚙️ How to Run the Code

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cross-camera-player-mapping.git
cd cross-camera-player-mapping
```

### 2. Install Dependencies
We recommend Python ≥ 3.9.
```bash
pip install -r requirements.txt
```

Main dependencies:
torch
torchvision
torchreid
ultralytics
scikit-learn
matplotlib
pandas

### 3. Download YOLO Weights
Download the YOLOv8 model from:

```bash
https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVCMD/view
```
Place the weights file (e.g. yolov8n.pt) into the repo directory.

### 4. Run Detection
Run detection on both videos to produce player crops:

```bash
python scripts/detect_players.py --video broadcast.mp4 --output_dir detections_broadcast
python scripts/detect_players.py --video tacticam.mp4 --output_dir detections_tacticam
```

### 5. Extract Features
Generate feature embeddings for each crop:

```bash
python scripts/extract_features.py
```

### 6. Match Players
Run matching to produce the similarity matrix and generate matched pairs:

```bash
python scripts/match_players.py
```
Results will be saved as:
matched_players.pkl
matched_players.csv

### 7. Visualize Matches (Optional)
To plot matched pairs side-by-side:

```bash
python scripts/visualize_matches.py
```

### Output Example:
Sample from matched_players.csv:
| tacticam_filename | broadcast_filename | similarity |
| ------------------ | ------------------- | ---------- |
| player_185.jpg    | player_103.jpg     | 0.7558     |
| player_18.jpg     | player_22.jpg      | 0.8031     |
| frame_195.jpg     | frame_31.jpg       | 0.7029     |

### Approach & Methodology:
Object Detection: YOLOv8 model detects player bounding boxes.
Feature Extraction: Player crops passed through a ReID backbone (e.g. OSNet) to extract feature embeddings.
Similarity Computation: Cosine similarity measures visual similarity between crops.
Matching: For each player in tacticam, the broadcast crop with the highest similarity is selected.
Visualization: Side-by-side images confirm visual matching quality.

### Challenges & Limitations:
Player Occlusions: Partial visibility affects detection and feature quality.
Appearance Variance: Different lighting or angles can reduce feature similarity.
Temporal Tracking: Currently matches frames individually rather than tracking players over time.

### Future Work
Integrate temporal tracking (e.g. DeepSort) to improve player identity stability.
Incorporate jersey number recognition as an additional matching cue.
Optimize runtime for real-time applications.

### License
MIT License
