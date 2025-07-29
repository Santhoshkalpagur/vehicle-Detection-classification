import os
from ultralytics import YOLO

# Define the paths
data_path = r'D:\New project(14-06-24)\VehiclesDetectionDataset\dataset.yaml'
checkpoint_path = r'runs\detect\train2\weights\last.pt'  # Path to your checkpoint file

# Load the model from the checkpoint
model = YOLO(checkpoint_path)

# Define the augmentation settings
augmentation_settings = {
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.5,
    'perspective': 0.001,
    'flipud': 0.5,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.2
}

# Resume training
model.train(data='dataset.yaml', epochs=120, **augmentation_settings)  # Adjust epochs as needed
