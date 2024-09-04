import numpy as np  
import torch
from ultralytics import YOLO, RTDETR
from argparse import ArgumentParser

random_seed = 2024
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
np.random.seed(random_seed)

if __name__ == '__main__':
     
    parser = ArgumentParser(description='Hyperparameters')

    parser.add_argument('--bs', nargs='?', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', nargs='?', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--aug', nargs='?', type=bool, default=False,
                        help='Whether to use data augmentation.')
    parser.add_argument('--model_scale', nargs='?', type=str, default='ela',
                        help='model scale type')
    parser.add_argument('--name', nargs='?', type=str, default='yolov8',
                        help='name')
    args = parser.parse_args()

    if args.model_scale == 'gam':
       model = YOLO('ultralytics/cfg/models/v8/yolov8s_GAM.yaml').load('/content/drive/My Drive/MSc Project/faulty_insulator/yolov8s_100aug_True/train_val_wo_sahi/weights/best.pt')

    elif args.model_scale == 'res-cbam':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_ResBlock_CBAM.yaml').load('/content/drive/My Drive/MSc Project/faulty_insulator/yolov8s_100aug_True/train_val_wo_sahi/weights/best.pt')

    elif args.model_scale == 'ela':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_ELA.yaml').load('/content/drive/My Drive/MSc Project/faulty_insulator/yolov8s_100aug_True/train_val_wo_sahi/weights/best.pt')
    
    elif args.model_scale == 'sa':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_SA.yaml').load('/content/drive/My Drive/MSc Project/faulty_insulator/yolov8s_100aug_True/train_val_wo_sahi/weights/best.pt')
    
    elif args.model_scale == 'eca':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s_ECA.yaml').load('/content/drive/My Drive/MSc Project/faulty_insulator/yolov8s_100aug_True/train_val_wo_sahi/weights/best.pt')
    
    elif args.model_scale == 'yolov9':
        model = YOLO("https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt")

    elif args.model_scale == 'yolov10':
        model = YOLO("https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt")

    elif args.model_scale == 'rt-detr':
        model = RTDETR(model='https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-x.pt')

    model.train(
        data='/content/drive/My Drive/MSc Project/faulty_insulator/complete_ds_v2.yaml',
        task='detect',
        mode='train',
        name=args.name+'_'+args.model_scale+str(args.epochs)+'aug_'+str(args.aug),
        epochs=args.epochs,
        batch=args.bs,
        imgsz=640,
        overlap_mask=True,
        save=True,
        optimizer='SGD',
        exist_ok=True,
        val=True,
        augment=args.aug,
        boxes=False,
        patience=15,
        plots=True,
        fliplr= 0.5,
        flipud=0.5,
        shear=0.5,
        translate=0.4,
        mosaic= .5, #you can modify
        mixup= 0.15, #you can modify
        copy_paste= 0.3, #you can modify,
        scale=0.9, #you can modify
    )

