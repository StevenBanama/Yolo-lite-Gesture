import os
import re
import cv2
import numpy as np
from utils.utils import image_preporcess, postprocess_boxes, nms, draw_bbox, build_params
from easydict import EasyDict as easydict
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from train import build_model

def run_result(models, org_img, input_size, class_num=8):
    original_image_size = org_img.shape[:2]
    img = image_preporcess(np.copy(org_img), [input_size, input_size])
    pred_mbbox, pred_lbbox = models.predict(np.array([img]))
    pred_bbox = np.concatenate([
        np.reshape(pred_mbbox, (-1, 5 + class_num)),
        np.reshape(pred_lbbox, (-1, 5 + class_num))], axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = nms(bboxes, 0.3, method='nms')

    for bb in bboxes:
        bb = bb.astype(np.int)
        b1, b2 = tuple(bb[:2]), tuple(bb[2:4])

        cv2.putText(org_img, "%s"%(bb[5]), (bb[0], bb[1]), 2, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(org_img, b1, b2, (0, 0, 255), 2)
    return bboxes


def run_test(params):
    models = build_model(params)

    org_img = img = cv2.imread("./data/test/344450.jpg")
    input_size = 224
    run_result(models, org_img, input_size, params.class_num)
    cv2.imwrite("test_gg.jpg", org_img)

def run_batch(params=build_params(True, True)):
    result = []
    models = build_model(params)

    for root, _, files in os.walk("./data/test"):
        for f in files:
            fpath = os.path.join(root, f)
            org_img = img = cv2.imread(fpath)
            input_size = params.test_input
            img_id = int(re.search("\d+", fpath).group(0))
            bboxes = run_result(models, org_img, input_size, params.class_num)
            for bb in bboxes:
                result.append([img_id, int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]), int(bb[4]), int(bb[5])]) 
    return np.array(result)


def freezon_graph(params):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    model = build_model(params)
    sess = tf.compat.v1.keras.backend.get_session()
    graph = sess.graph
    model_name = "test.pb"
    outdir = "./test/"
    with graph.as_default():
        with sess.as_default():
            out_names = [v.name.split(":")[0] for v in model.output]
            node_names = out_names
            frozon_graph = convert_variables_to_constants(sess, graph.as_graph_def(), node_names)
            tf.compat.v1.train.write_graph(frozon_graph, outdir, model_name, as_text=False)

def video(params):
    cap = cv2.VideoCapture(0)
    models = build_model(params)
    input_size = 224

    while(True):
        ret, org_img = cap.read()
        if not ret:
            break
        original_image_size = org_img.shape[:2]

        run_result(models, org_img, input_size, params.class_num)

        cv2.imshow("camera", org_img)


if __name__ == "__main__":
    params = build_params(True, True)
    freezon_graph(params)
    if params.mode == "test":
        run_test(params)
    else if params.mode == "freeze":
        run_test(params)
    else:
        video(params)
