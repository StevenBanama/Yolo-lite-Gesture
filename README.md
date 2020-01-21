# gesture detector
     This is a sample code about yolo-lite.

# reqiurments
   >> pip install -r requirement.txt 

# train
    default parameter use "python train -h"
   >> python train.py --mode train --pretrain_model=./pretrained/cp-30-3.614092 --se --bn

# test
   python evaluate.py --pretrain_model=./pretrained/cp-30-3.614092 --se --bn # hack api depents on coco-tools 

# video
   >> python demo.py --mode video --pretrain_model=./pretrained/cp-30-3.614092 --se --bn
   >> python demo.py --mode video --pretrain_model=./pretrained/cp-145-4.073046

# to-do 
    1. coco-pretrain model
    2. some serilization-bug between keras and tf2.1
       cant load model with partial. (as far known)
    3. other bug 

# refrence
   https://arxiv.org/pdf/1811.05588
   https://towardsdatascience.com/review-senet-squeeze-and-excitation-network-winner-of-ilsvrc-2017-image-classification-a887b98b2883
   https://github.com/YunYang1994/tensorflow-yolov3
   https://github.com/cocodataset/cocoapi/blob/e140a084d678eacd18e85a9d8cfa45d1d5911db9/PythonAPI/pycocotools/coco.py
