from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model (person class)
model = YOLO('yolov8n.pt')  # or yolov8s.pt for better accuracy

# Input video
video_path = 'broadcast.mp4'
cap = cv2.VideoCapture(video_path)

frame_id = 0
output_dir = 'detections_broadcast'
os.makedirs(output_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Filter person detections (class 0 in COCO is 'person')
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            # Save cropped player image
            cv2.imwrite(f"{output_dir}/player_{frame_id}.jpg", crop)

            # (Optional) draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # (Optional) save visualized frame
    cv2.imwrite(f"{output_dir}/frame_{frame_id}.jpg", frame)
    frame_id += 1

cap.release()
