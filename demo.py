import os
import re
import cv2
import numpy as np
from utils.utils import image_preporcess, postprocess_boxes, nms, draw_bbox, build_params, tcost
from easydict import EasyDict as easydict
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from train import build_model


def checkpoint_loader(params):
    models = build_model(params)

    @tcost
    def run_result(org_img, input_size, params):
        original_image_size = org_img.shape[:2]
        img = image_preporcess(np.copy(org_img), [input_size, input_size], canny=params.canny)
        pred_mbbox, pred_lbbox = models.predict(np.array([img]))
        pred_bbox = np.concatenate([
            np.reshape(pred_mbbox, (-1, 5 + params.class_num)),
            np.reshape(pred_lbbox, (-1, 5 + params.class_num))], axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
        bboxes = nms(bboxes, 0.3, method='nms')

        for bb in bboxes:
            bb = bb.astype(np.int)
            b1, b2 = tuple(bb[:2]), tuple(bb[2:4])
            cate_id = bb[5]

            cv2.putText(org_img, "%s"%(params.id2cate.get(cate_id, cate_id)), (bb[0], bb[1]), 2, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(org_img, b1, b2, (0, 0, 255), 2)
        return bboxes
    return run_result

def pb_loader(params):
    import tensorflow as tf

    graph = tf.Graph() 
    pb_file = "./test/test.pb"
    return_elements = ["input_1:0", "branch/mid:0", "branch/large:0"]
    rtensor = read_pb_return_tensors(graph, pb_file, return_elements)
    print(rtensor)
    sess = tf.compat.v1.Session(graph=graph)

    @tcost
    def run_result(org_img, input_size, params):
        original_image_size = org_img.shape[:2]
        img = image_preporcess(np.copy(org_img), [input_size, input_size], canny=params.canny)
        pred_mbbox, pred_lbbox = sess.run(rtensor[1:], feed_dict={rtensor[0]: [img]})
        pred_bbox = np.concatenate([
            np.reshape(pred_mbbox, (-1, 5 + params.class_num)),
            np.reshape(pred_lbbox, (-1, 5 + params.class_num))], axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
        bboxes = nms(bboxes, 0.3, method='nms')

        for bb in bboxes:
            bb = bb.astype(np.int)
            b1, b2 = tuple(bb[:2]), tuple(bb[2:4])
            cate_id = bb[5]

            cv2.putText(org_img, "%s"%(params.id2cate.get(cate_id, cate_id)), (bb[0], bb[1]), 2, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(org_img, b1, b2, (0, 0, 255), 2)
        return bboxes


    return run_result


def model_loader(params):
    if params.pretrain_model.find("pb") != -1:
        return pb_loader(params)
    else:
        return checkpoint_loader(params) 

def run_test(params):
    proc = model_loader(params)

    org_img = img = cv2.imread("./data/test/344450.jpg")
    input_size = params.test_input
    proc(org_img, input_size, params)
    cv2.imwrite("test_gg.jpg", org_img)

def run_batch(params):
    result = []
    proc = model_loader(params) 

    for root, _, files in os.walk("./data/test"):
        for f in files:
            fpath = os.path.join(root, f)
            org_img = img = cv2.imread(fpath)
            input_size = params.test_input
            img_id = int(re.search("\d+", fpath).group(0))
            bboxes = proc(org_img, input_size, params)
            for bb in bboxes:
                result.append([img_id, int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]), int(bb[4]), int(bb[5])]) 
    return np.array(result)


def freezon_graph(params):
    # from tensorflow.python.framework.graph_util import convert_variables_to_constants
    # https://github.com/tensorflow/tensorflow/issues/31331
    import tensorflow as tf
    import tensorflow.compat.v1.keras.backend as K
    from tensorflow.python.tools import freeze_graph

    K.set_learning_phase(0)

    model = build_model(params)
    
    sess = K.get_session()
    graph = sess.graph
    pb_file = "test.pb"
    outdir = "./test/"

    tf.compat.v1.saved_model.simple_save(K.get_session(),
        outdir,
        inputs={"input": model.inputs[0]},
        outputs={"output0": model.outputs[0], "output1": model.outputs[1]})

    freeze_graph.freeze_graph(None,
        None,
        None,
        None,
        model.outputs[0].op.name + "," + model.outputs[1].op.name,
        None,
        None,
        os.path.join(outdir, pb_file),
        False,
        "",
        input_saved_model_dir=outdir)

def read_pb_return_tensors(graph, pb_file, return_elements):
    import tensorflow as tf
    from tensorflow.compat.v1.keras import backend as K

    with tf.compat.v1.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.compat.v1.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements, name="")
    return return_elements


def video(params):
    cap = cv2.VideoCapture(0)
    proc = model_loader(params)
    input_size = params.test_input

    while(True):
        ret, org_img = cap.read()
        if not ret:
            break
        original_image_size = org_img.shape[:2]

        proc(org_img, input_size, params)

        cv2.imshow("camera", org_img)
        cv2.waitKey(1)


if __name__ == "__main__":
    params = build_params()
    if params.mode == "test":
        run_test(params)
    elif params.mode == "freeze":
        # freezon_graph(params)
        pb_loader(params)
    elif params.mode == "batch":
        run_batch(params)
    else:
        video(params)
