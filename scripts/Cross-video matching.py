import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load feature files
# -------------------------------

with open('features_broadcast.pkl', 'rb') as f:
    broadcast_features = pickle.load(f)

with open('features_tacticam.pkl', 'rb') as f:
    tacticam_features = pickle.load(f)

print(f"Loaded {len(broadcast_features)} broadcast features.")
print(f"Loaded {len(tacticam_features)} tacticam features.")

# -------------------------------
# Prepare feature arrays
# -------------------------------

broadcast_matrix = np.array([item['feature'] for item in broadcast_features])
broadcast_filenames = [item['filename'] for item in broadcast_features]

tacticam_matrix = np.array([item['feature'] for item in tacticam_features])
tacticam_filenames = [item['filename'] for item in tacticam_features]

# -------------------------------
# Compute similarity
# -------------------------------

similarity_matrix = cosine_similarity(tacticam_matrix, broadcast_matrix)

# -------------------------------
# Find best matches
# -------------------------------

threshold = 0.5     # tweak this higher for stricter matches

matches = []

for i, row in enumerate(similarity_matrix):
    best_idx = np.argmax(row)
    best_score = row[best_idx]

    if best_score >= threshold:
        matches.append({
            'tacticam_filename': tacticam_filenames[i],
            'broadcast_filename': broadcast_filenames[best_idx],
            'similarity': float(best_score)
        })

print(f"\n Found {len(matches)} matches above threshold {threshold}.")

for m in matches:
    print(f"Tacticam: {m['tacticam_filename']} --> Broadcast: {m['broadcast_filename']} "
          f"Similarity: {m['similarity']:.3f}")

# -------------------------------
# Save matches
# -------------------------------

with open('matched_players.pkl', 'wb') as f:
    pickle.dump(matches, f)

print("\n Matches saved to matched_players.pkl!")
