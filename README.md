# gesture detector
     This is a sample code about yolo-lite.

# reqiurments
   pip install -r requirement.txt 

# train
   // default parameter use "python train -h"
   python train.py

# test
   python evaluate.py  # hack api depents on coco-tools
   
   unstable stats depents on only few sample (input=224, 40ms/per roi).
   |-|
   | Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.476 |
   | Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.884 |
   | Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.465 |
   | Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000 |
   | Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.369 |
   | Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.527 |
   | Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.544 |
   | Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.544 |
   | Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.544 |
   | Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000 |
   | Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.436 |
   | Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.587 |

# video
   python demo.py --mode video

# to-do 
    1. coco-pretrain model
    2. some serilization-bug between keras and tf
       cant load model with partial. (as far known)
    3. other bug 

# refrence
   https://arxiv.org/pdf/1811.05588
   https://github.com/YunYang1994/tensorflow-yolov3
   https://github.com/cocodataset/cocoapi/blob/e140a084d678eacd18e85a9d8cfa45d1d5911db9/PythonAPI/pycocotools/coco.py
