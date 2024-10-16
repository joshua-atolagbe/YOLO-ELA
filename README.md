# YOLO-ELA: Efficient Local Attention Modeling for High-Performance Real-Time Insulator Defect Detection

## Architecture
<img src='images/arc.png'>

## Dataset
- Training and validation data can be downloaded [here](https://drive.google.com/file/d/1-_A4Oi-Hg6dT4y6uSZLk7ey-zfJKPOVm/view)
- Test data can be downloaded [here](https://kaggle.com/competitions/innopolis-high-voltage-challenge). 

## Requirements
Install requirements
```shell
pip install -r requirements.txt
```
+ +NVIDIA GPU + CUDA CuDNN
+ +Linux (Ubuntu)
+ +Python 3.10

 
## Get the test results
- Download YOLO-ELA checkpoint [here](https://drive.google.com/drive/folders/1-jFe_q_AVnc6-5tWAvlxKJDnIycE2vTY?usp=sharing)
- Open a terminal and run
```shell
python test.py\
        --model 'models/yolo_ela.pt'\ 
       
```
This automatically create a new directory called `run`. Navigate to see results
<img src='images/result.png'>


## How to train YOLO-ELA 

- Open terminal and run
```shell
python train.py \
        --model_scale 'ela' \
        --cfg 'ultralytics/cfg/data.yaml' \
        --aug True \
        --name 'ela' \
        --epochs 100 \
        --bs 16 \
        --img_sz 640 # Image size can either be 320 or 640

```
## Citing
```python
@article{yoloela,
  author = {Olalekan Akindele and Joshua Atolagbe},
  title = {YOLO-ELA: Efficient Local Attention Modeling for High-Performance Real-Time Insulator Defect Detection},
  journal = {arXiv preprint arXiv:2410.11727},
  year = {2024}
}
```
## Credit
The codes in this repository are based on [Ultralytics](https://github.com/ultralytics/ultralytics)

