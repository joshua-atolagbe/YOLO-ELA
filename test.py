import numpy as np  
import torch
from ultralytics import YOLO
from argparse import ArgumentParser

random_seed = 2024
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
np.random.seed(random_seed)

if __name__ == '__main__':
     
    parser = ArgumentParser(description='Hyperparameters')

    parser.add_argument('--model', nargs='?', type=str, default='yolo_ela.pt',
                        help='model weight')
                        
    args = parser.parse_args()
    
    model = YOLO(args.model)

    model.val(
        data='ultralytics/cfg/data.yaml',
        conf=0.3,
        iou=0.6,
        split='test'
        imgsz=3008
    )
