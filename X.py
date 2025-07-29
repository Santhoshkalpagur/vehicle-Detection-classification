import os
from ultralytics import YOLO
import cv2
import pandas as pd

VIDEOS_DIR = os.path.join(r"D:\New project(14-06-24)\VehiclesDetectionDataset")

video_path = os.path.join(VIDEOS_DIR, '3v.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = r'D:\New project(14-06-24)\VehiclesDetectionDataset\runs\detect\train14\weights\last.pt'
model = YOLO(model_path)  # load a custom model

threshold = 0.3

# Initialize a list to store detection information
detections = []

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Save detection information
            detections.append({
                'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
                'score': float(score),
                'class_id': int(class_id),
                'class_name': results.names[int(class_id)].upper()
            })

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

# Convert detections to a DataFrame and save to an Excel file
df = pd.DataFrame(detections)
excel_path = os.path.join(VIDEOS_DIR, 'detections.xlsx')
df.to_excel(excel_path, index=False)

print(f'Detections saved to {excel_path}')
