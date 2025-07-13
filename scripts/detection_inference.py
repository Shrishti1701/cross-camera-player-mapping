"""
detection_inference.py

Run YOLOv8 object detection on a video and save cropped player images.
"""

from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Change video_path and output_dir as needed
video_path = 'broadcast.mp4'
output_dir = 'detections_broadcast'

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            crop_filename = f"{output_dir}/player_{frame_id}.jpg"
            cv2.imwrite(crop_filename, crop)
            print(f"Saved crop: {crop_filename}")

    frame_id += 1

cap.release()
print("✅ YOLO detection finished!")
