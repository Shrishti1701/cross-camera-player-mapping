from ultralytics import YOLO
import cv2
import os

# -------------------------------
# Video list
# -------------------------------
videos = [
    {
        "video_path": "broadcast.mp4",
        "output_dir": "detections_broadcast"
    },
    {
        "video_path": "tacticam.mp4",
        "output_dir": "detections_tacticam"
    }
]

# Load YOLO model once
model = YOLO('yolov8n.pt')   # or yolov8s.pt for better accuracy

for video in videos:
    video_path = video["video_path"]
    output_dir = video["output_dir"]

    cap = cv2.VideoCapture(video_path)

    os.makedirs(output_dir, exist_ok=True)

    frame_id = 0

    print(f"🔎 Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        person_found = False

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # Class 0 = person in COCO
                person_found = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crop_filename = f"{output_dir}/player_{frame_id}.jpg"
                cv2.imwrite(crop_filename, crop)

                print(f"✅ Saved crop: {crop_filename}")

        if person_found:
            frame_filename = f"{output_dir}/frame_{frame_id}.jpg"
            cv2.imwrite(frame_filename, frame)

        frame_id += 1

    cap.release()
    print(f" Done processing {video_path}\n")

print(" YOLO processing complete for ALL videos.")
