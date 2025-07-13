"""
visualize_matches.py

Plot matched player crops side by side for visual verification.
"""

import pickle
import os
import cv2
import matplotlib.pyplot as plt

# Load matched results
with open('matched_players.pkl', 'rb') as f:
    matches = pickle.load(f)

# Show top N matches
N = 10

for match in matches[:N]:
    tacticam_path = os.path.join("detections_tacticam", match["tacticam_filename"])
    broadcast_path = os.path.join("detections_broadcast", match["broadcast_filename"])

    img1 = cv2.imread(tacticam_path)
    img2 = cv2.imread(broadcast_path)

    if img1 is None or img2 is None:
        print(f"Missing image: {match}")
        continue

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img1)
    axes[0].set_title(match["tacticam_filename"])
    axes[1].imshow(img2)
    axes[1].set_title(match["broadcast_filename"])

    for ax in axes:
        ax.axis("off")

    plt.show()
