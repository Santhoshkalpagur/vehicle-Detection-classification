import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from ultralytics import YOLO

# Define paths
test_images_dir = r'D:\New project(14-06-24)\VehiclesDetectionDataset\images\test'
test_labels_dir = r'D:\New project(14-06-24)\VehiclesDetectionDataset\labels\test'
model_path = r'D:\New project(14-06-24)\VehiclesDetectionDataset\runs\detect\train14\weights\last.pt'

# Load the YOLOv8 model
model = YOLO(model_path)


# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


# Function to read labels from txt files
def read_labels(label_path):
    with open(label_path, 'r') as file:
        labels = [int(line.strip().split()[0]) for line in file]
    return labels


# Initialize lists to store true and predicted labels
y_true = []
y_pred = []

# Get list of test images
test_images = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Iterate over each test image
for image_filename in test_images:
    image_path = os.path.join(test_images_dir, image_filename)
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_path = os.path.join(test_labels_dir, label_filename)

    # Preprocess the image
    image = preprocess_image(image_path)

    # Read true labels
    true_labels = read_labels(label_path)

    # Perform inference
    results = model(image)

    # Ensure results are in the expected format (list of Results objects)
    if not isinstance(results, list):
        results = [results]

    # Extract detections and their labels
    for result in results:
        try:
            # Assuming 'boxes' attribute exists for bounding boxes
            detections = result.boxes[0].cpu().numpy()

            # Iterate through detections
            for detection in detections:
                predicted_class = int(detection[5])  # Assuming class is at index 5

                # Add true and predicted labels to lists
                if true_labels:
                    y_true.append(true_labels[0])  # Assuming one label per image
                else:
                    y_true.append(-1)  # Handle cases where no label is available

                y_pred.append(predicted_class)

        except AttributeError:
            raise AttributeError(f"Error: Unable to access 'boxes' attribute in 'Results' object.")
        except IndexError:
            print(f"Warning: No detections found in {image_filename}")

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

# Display evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Plot confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
