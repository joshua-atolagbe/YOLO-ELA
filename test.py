# Import necessary libraries
import numpy as np  
import torch
from ultralytics import YOLO
from argparse import ArgumentParser

# Set a random seed for reproducibility across multiple runs
random_seed = 2024
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)  # Set CUDA seeds for all devices (if available)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior with cuDNN
np.random.seed(random_seed)

if __name__ == '__main__':
    # Initialize the argument parser to handle command-line arguments
    parser = ArgumentParser(description='Hyperparameters')

    # Define command-line argument for specifying the model's weight file
    parser.add_argument('--model', nargs='?', type=str, default='yolo_ela.pt',
                        help='Path to the model weight file')
                        
    args = parser.parse_args()
    
    # Load the YOLO model with the specified weights
    model = YOLO(args.model)

    # Validate the model on a dataset
    model.val(
        data='ultralytics/cfg/data.yaml',  # Path to the dataset configuration file
        conf=0.3,                          # Confidence threshold for detections
        iou=0.6,                           # Intersection Over Union (IOU) threshold for non-max suppression
        split='test',                      # Dataset split to use for validation (e.g., 'test', 'val')
        imgsz=3008                         # Size of input images for validation
    )
