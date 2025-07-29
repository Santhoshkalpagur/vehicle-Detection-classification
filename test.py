import os
from ultralytics import YOLO
import cv2

# Define the image file path
IMAGE_FILE = r'D:\New project(14-06-24)\VehiclesDetectionDataset\4.jpg'

# Verify if the image file exists
if not os.path.exists(IMAGE_FILE):
    print(f"Error: Image file {IMAGE_FILE} does not exist.")
    exit()

# Read the image
image = cv2.imread(IMAGE_FILE)

# Load the YOLO model
model_path = r'D:\New project(14-06-24)\VehiclesDetectionDataset\runs\detect\train14\weights\last.pt'
model = YOLO(model_path)  # load a custom model

# Perform object detection on the image
results = model(image)

# Check if results is a list (indicating batch processing)
if isinstance(results[0], list):
    # Process each image's detections in batch mode
    for idx, detections in enumerate(results):
        print(f"Results for image {idx}:")
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det[:6]
            if isinstance(conf, float) and conf > 0.3:  # Check if conf is a float and above threshold
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, model.names[int(class_id)], (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
else:
    # Process detections for a single image
    for det in results:
        x1, y1, x2, y2, conf, class_id = det[:6]
        if isinstance(conf, float) and conf > 0.3:  # Check if conf is a float and above threshold
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, model.names[int(class_id)], (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

# Display the result
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

