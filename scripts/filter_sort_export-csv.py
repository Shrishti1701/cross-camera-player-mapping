### FILTER MATCHES BY THRESHOLD
filtered_matches = [m for m in matches if m["similarity"] > 0.7]
print(f"Remaining matches after threshold: {len(filtered_matches)}")

### SORT SIMILARITY
matches_sorted = sorted(matches, key=lambda m: m["similarity"], reverse=True)

### EXPORT CSV FOR EASY INSPECTION
import pickle
import pandas as pd

# Load matched pairs
with open("matched_players.pkl", "rb") as f:
    matches = pickle.load(f)

print(f"Loaded {len(matches)} matches.")

# Convert to DataFrame
df = pd.DataFrame(matches)

# Save to CSV
csv_filename = "matched_players.csv"
df.to_csv(csv_filename, index=False)
print(f" CSV saved to {csv_filename}")

### CONTENTS OF CSV
import pandas as pd

df = pd.read_csv("matched_players.csv")
print(df.head())
