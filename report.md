# Cross-Camera Player Mapping - Report

## Objective

Given two clips (`broadcast.mp4` and `tacticam.mp4`) of the same sports game from different camera angles, the goal was to:

- Detect players in both videos.
- Extract visual features for each detected player.
- Map each player from the tacticam view to their corresponding identity in the broadcast view so that player IDs remain consistent across both feeds.

---

## Approach & Methodology

### 1. Object Detection

- **Model:** YOLOv8 (fine-tuned version provided in the assignment).
- **Task:** Detect all players in each frame of both videos.
- **Output:** Player bounding box crops saved as JPG images in:
  - `detections_broadcast/`
  - `detections_tacticam/`

---

### 2. Feature Extraction

- Used a person re-identification (ReID) backbone (OSNet via `torchreid`) to extract feature embeddings for each cropped player image.
- Each player image converted into a 512-dimensional embedding vector.
- Saved embeddings into:
  - `features_broadcast.pkl`
  - `features_tacticam.pkl`

---

### 3. Player Matching

- Computed pairwise cosine similarity between all broadcast and tacticam embeddings.
- For each tacticam player crop, selected the broadcast crop with the highest similarity score as the best match.
- Saved results in:
  - `matched_players.pkl`
  - `matched_players.csv`

---

### 4. Visualization

- Created side-by-side plots of matched player crops to visually confirm mapping quality.
- Provided an optional script (`visualize_matches.py`) to display matches from the CSV.

---

## Techniques Tried & Outcomes

- **YOLOv8 detection:** Successful in detecting players, though sometimes partial crops occurred due to occlusion.
- **ReID embedding comparison:** Worked well for visually similar players across views, achieving many matches with similarity >0.75.
- **Cosine similarity matching:** Effective, but sensitivity observed for lighting and angle differences.

---

## Challenges Encountered

- **Occlusions & Partial Detections:** Players partially visible in some frames resulted in lower similarity scores.
- **Appearance Variance:** Different lighting conditions and camera angles reduced feature similarity for some players.
- **Temporal Tracking:** The current pipeline matches frames independently rather than tracking players across frames over time.

---

## Future Work

If I had more time/resources, I would:

- Integrate a temporal tracking algorithm (e.g. DeepSORT) to maintain player identities across frames.
- Incorporate jersey number recognition as an additional cue for matching.
- Fine-tune ReID models specifically for sports player appearances to improve robustness.
- Optimize for speed to support near-real-time inference.

---

## Completion Status

✅ All main objectives achieved:
- Player detection performed successfully on both videos.
- Feature embeddings extracted for all detected players.
- Matching between tacticam and broadcast views completed and results saved.

No major incomplete sections remain. Further improvements could be made for robustness and tracking, but the project fulfills the assignment requirements as specified.

---

## Files Included

- `scripts/` – all Python scripts for detection, feature extraction, matching, and visualization.
- `matched_players.pkl` – Pickle file with matching results.
- `matched_players.csv` – CSV export of all matched pairs.
- Sample detections in:
  - `detections_broadcast/`
  - `detections_tacticam/`

