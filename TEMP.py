import os
from ultralytics import YOLO
import cv2
import pandas as pd

# Directory where the Excel sheet will be saved
VIDEOS_DIR = os.path.join(r"D:\New project(14-06-24)\VehiclesDetectionDataset")

# Initialize capture from USB camera (assuming it's the first device, adjust if necessary)
cap = cv2.VideoCapture(0)  # Use 0, 1, 2, etc. for different cameras

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open USB camera.")
    exit()

# Get frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Failed to read frame from USB camera.")
    exit()
H, W, _ = frame.shape

# Video writer setup
video_path_out = os.path.join(VIDEOS_DIR, 'usb_camera_out.mp4')
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))

# Model setup
model_path = r'D:\New project(14-06-24)\VehiclesDetectionDataset\runs\detect\train14\weights\last.pt'
model = YOLO(model_path)  # load a custom model

# Detection threshold
threshold = 0.3

# Initialize a list to store detection information
detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from USB camera.")
        break

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

    # Write frame with detections to output video
    out.write(frame)

    # Display frame
    cv2.imshow('USB Camera Feed', frame)

    # Check for 'q' key to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Convert detections to a DataFrame and save to an Excel file
df = pd.DataFrame(detections)
excel_path = os.path.join(VIDEOS_DIR, 'detections_usb_camera.xlsx')
df.to_excel(excel_path, index=False)

print(f'Detections saved to {excel_path}')
