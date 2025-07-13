"""
extract_features.py

Extract feature embeddings from cropped player images using Torchreid.
"""

import torchreid
import os
import cv2
import torch
import pickle
from torchvision import transforms as T

# Configure paths
crop_dir = 'detections_broadcast'
output_pickle = 'features_broadcast.pkl'

# Load Torchreid model
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

features = []

for filename in os.listdir(crop_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(crop_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
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

with open(output_pickle, 'wb') as f:
    pickle.dump(features, f)

print(f" Saved features to {output_pickle}")
