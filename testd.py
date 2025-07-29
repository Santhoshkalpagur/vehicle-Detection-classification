import os
import cv2
from ultralytics import YOLO

# Directory containing the images
IMAGES_DIR = r"D:\New project(14-06-24)\VehiclesDetectionDataset"

# Output directory for annotated images
OUTPUT_DIR = os.path.join(IMAGES_DIR, 'annotated_images')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of image files to process
image_files = [
    os.path.join(IMAGES_DIR, '5.jpeg'),
    # os.path.join(IMAGES_DIR, '2.jpg'),
    # os.path.join(IMAGES_DIR, '3.jpg'),
    # Add more image paths as needed
]

# Path to the YOLO model weights
model_path = r'D:\New project(14-06-24)\VehiclesDetectionDataset\runs\detect\train14\weights\last.pt'

# Load YOLO model
model = YOLO(model_path)

# Detection threshold
threshold = 0.3

# Process each image file
for image_path in image_files:
    # Load the image
    image = cv2.imread(image_path)

    # Perform object detection on the image
    results = model(image)[0]

    # Iterate through each detected object
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Draw bounding box and label if score exceeds threshold
        if score > threshold:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Save annotated image to output directory
    output_image_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)

    print(f"Processed: {image_path}")

print("All images processed.")
