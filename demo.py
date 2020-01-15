import os
import cv2
import numpy as np
from utils.utils import image_preporcess, postprocess_boxes, nms, draw_bbox, build_params
from easydict import EasyDict as easydict
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from yolo_lite_keras import build_model

def run_result(models, org_img, input_size):
    original_image_size = org_img.shape[:2]
    img = image_preporcess(np.copy(org_img), [input_size, input_size])
    pred_mbbox, pred_lbbox = models.predict(np.array([img]))
    pred_bbox = np.concatenate([
        np.reshape(pred_mbbox, (-1, 5 + params.class_num)),
        np.reshape(pred_lbbox, (-1, 5 + params.class_num))], axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = nms(bboxes, 0.3, method='nms')

    for bb in bboxes:
        bb = bb.astype(np.int)
        b1, b2 = tuple(bb[:2]), tuple(bb[2:4])

        cv2.putText(org_img, "%s"%(bb[5]), (bb[0], bb[1]), 2, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(org_img, b1, b2, (0, 0, 255), 2)
    print(bboxes)
    return org_img


def run_test(params):
    models = build_model(params)

    org_img = img = cv2.imread("./data/test/344450.jpg")
    input_size = 224
    run_result(models, org_img, input_size)
    cv2.imwrite("test_gg.jpg", org_img)

def video(params):
    cap = cv2.VideoCapture(0)
    models = build_model(params)
    input_size = 224

    while(True):
        ret, org_img = cap.read()
        if not ret:
            break
        original_image_size = org_img.shape[:2]

        run_result(models, org_img, input_size)

        cv2.imshow("camera", org_img)


if __name__ == "__main__":
    params = build_params(True, True)
    video(params)
