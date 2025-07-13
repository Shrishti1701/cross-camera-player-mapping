"""
compute_similarity.py

Compute cosine similarity between broadcast and tacticam embeddings,
and save the matched pairs into CSV and pickle.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load features
with open('features_broadcast.pkl', 'rb') as f:
    broadcast_features = pickle.load(f)

with open('features_tacticam.pkl', 'rb') as f:
    tacticam_features = pickle.load(f)

# Prepare matrices
broadcast_vectors = np.array([x["feature"] for x in broadcast_features])
tacticam_vectors = np.array([x["feature"] for x in tacticam_features])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(tacticam_vectors, broadcast_vectors)

# Prepare results
results = []
for i, tacticam_feat in enumerate(tacticam_features):
    best_idx = np.argmax(similarity_matrix[i])
    best_score = similarity_matrix[i, best_idx]
    broadcast_filename = broadcast_features[best_idx]["filename"]

    results.append({
        "tacticam_filename": tacticam_feat["filename"],
        "broadcast_filename": broadcast_filename,
        "similarity": best_score
    })

# Save as pickle
with open("matched_players.pkl", "wb") as f:
    pickle.dump(results, f)

# Save as CSV
df = pd.DataFrame(results)
df.to_csv("matched_players.csv", index=False)

print(" Player matching complete! Results saved.")
