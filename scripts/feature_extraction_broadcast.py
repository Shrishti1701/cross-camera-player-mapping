import torchreid
import os
import cv2
import numpy as np
import torch
from torchvision import transforms as T
import pickle

# -------------------------------
# Load Torchreid Model
# -------------------------------

model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# -------------------------------
# Define Transform
# -------------------------------

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Process Broadcast Folder
# -------------------------------

crop_dir = 'detections_broadcast'
output_pickle = 'features_broadcast.pkl'

features = []

for filename in os.listdir(crop_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(crop_dir, filename)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable file: {filename}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(img_tensor)

        feat = feat.cpu().numpy().flatten()

        features.append({
            'filename': filename,
            'feature': feat
        })

print(f" Extracted features for {len(features)} images.")

with open(output_pickle, 'wb') as f:
    pickle.dump(features, f)

print(f" Features saved to {output_pickle}")
