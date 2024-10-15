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

# URL for downloading a pre-trained YOLOv8 small model
v8_small = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt'

if __name__ == '__main__':
    # Initialize the argument parser to handle command-line arguments
    parser = ArgumentParser(description='Hyperparameters')

    # Define command-line arguments for model training configuration
    parser.add_argument('--bs', nargs='?', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', nargs='?', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--aug', nargs='?', type=bool, default=False,
                        help='Enable data augmentation if set to True')
    parser.add_argument('--img_sz', nargs='?', type=int, default=640,
                        help='Image size for training and validation')
    parser.add_argument('--model_scale', nargs='?', type=str, default='ela-large',
                        help='Model variant to use for training')
    parser.add_argument('--name', nargs='?', type=str, default='yolov8',
                        help='Project name for saving training results')
                        
    args = parser.parse_args()

    # Model selection based on the --model_scale argument
    if args.model_scale == 'gam':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_GAM.yaml').load(v8_small)
    elif args.model_scale == 'baseline':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml').load(v8_small)
    elif args.model_scale == 'cam':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_CAM.yaml').load(v8_small)
    elif args.model_scale == 'cbam':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_CBAM.yaml').load(v8_small)
    elif args.model_scale == 'rescbam':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_ResBlock_CBAM.yaml').load(v8_small)
    elif args.model_scale == 'ela':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_ELA.yaml').load(v8_small)
    elif args.model_scale == 'mlca':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_MLCA.yaml').load(v8_small)
    elif args.model_scale == 'sa':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_SA.yaml').load(v8_small)
    elif args.model_scale == 'eca':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_ECA.yaml').load(v8_small)
    elif args.model_scale == 'ca':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_CA.yaml').load(v8_small)

    # Train the selected model with specified parameters
    model.train(
        data='ultralytics/cfg/data.yaml',  # Path to the dataset configuration file
        task='detect',                     # Task type, here it's 'detect' for object detection
        mode='train',                      # Mode is set to 'train' for training
        name=f"{args.name}_{args.model_scale}_{args.epochs}_aug_{args.aug}",  # Project name for saving results
        epochs=args.epochs,                # Number of training epochs
        batch=args.bs,                     # Batch size
        imgsz=args.img_sz,                 # Image size for training
        overlap_mask=True,                 # Enable overlapping mask
        save=True,                         # Save model checkpoints and results
        optimizer='SGD',                   # Optimizer type, Stochastic Gradient Descent here
        exist_ok=True,                     # Allow existing project folder
        val=True,                          # Perform validation after training
        augment=args.aug,                  # Enable or disable data augmentation
        boxes=False,                       # Show boxes if True
        patience=50,                       # Early stopping patience
        plots=True,                        # Generate training plots
        fliplr=0.5,                        # Left-right flip augmentation probability
        flipud=0.5,                        # Up-down flip augmentation probability
        mosaic=0.5,                        # Mosaic augmentation (you can modify this)
        mixup=0.15,                        # Mixup augmentation (you can modify this)
        copy_paste=0.3,                    # Copy-paste augmentation (you can modify this)
        scale=0.5,                         # Scale augmentation factor (you can modify this)
    )
