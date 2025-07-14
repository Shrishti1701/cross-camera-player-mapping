import os
import pickle
import matplotlib.pyplot as plt
import cv2

# Paths
tac_dir = "detections_tacticam"
bro_dir = "detections_broadcast"

# Load matches
with open("matched_players.pkl", "rb") as f:
    matches = pickle.load(f)

print(f"Loaded {len(matches)} matches.")

for m in matches[:10]:  # test only first 10
    tac_filename = m["tacticam_filename"]
    bro_filename = m["broadcast_filename"]

    tac_path = os.path.join(tac_dir, tac_filename)
    bro_path = os.path.join(bro_dir, bro_filename)

    print(f"Tacticam → {tac_path}  Exists: {os.path.exists(tac_path)}")
    print(f"Broadcast → {bro_path}  Exists: {os.path.exists(bro_path)}")

    # Check that paths exist
    if not os.path.exists(tac_path) or not os.path.exists(bro_path):
        print("Skipping because file not found.\n")
        continue

    # Load images
    img_tac = cv2.imread(tac_path)
    img_bro = cv2.imread(bro_path)

    if img_tac is None:
        print(f"❌ Could not read tacticam image: {tac_path}")
        continue

    if img_bro is None:
        print(f"❌ Could not read broadcast image: {bro_path}")
        continue

    # Convert BGR to RGB for matplotlib
    img_tac = cv2.cvtColor(img_tac, cv2.COLOR_BGR2RGB)
    img_bro = cv2.cvtColor(img_bro, cv2.COLOR_BGR2RGB)

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_tac)
    axes[0].set_title(f"Tacticam\n{tac_filename}")

    axes[1].imshow(img_bro)
    axes[1].set_title(f"Broadcast\n{bro_filename}")

    for ax in axes:
        ax.axis("off")

    plt.show()
