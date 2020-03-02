import os
import re
import cv2
import numpy as np
from utils.utils import image_preporcess, postprocess_boxes, nms, draw_bbox, build_params, tcost
from easydict import EasyDict as easydict
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
from train import build_model


def draw_boxes(params, org_img, bboxes):
    for bb in bboxes:
        bb = bb.astype(np.int)
        b1, b2 = tuple(bb[:2]), tuple(bb[2:4])
        cate_id = bb[5]

        cv2.putText(org_img, "%s"%(params.id2cate.get(cate_id, cate_id)), (bb[0], bb[1]), 2, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(org_img, b1, b2, (0, 0, 255), 2)


def checkpoint_loader(params):
    models = build_model(params)

    @tcost
    def run_result(org_img, input_size, params):
        original_image_size = org_img.shape[:2]
        img = image_preporcess(np.copy(org_img), [input_size, input_size], canny=params.canny)
        pred_mbbox, pred_lbbox = models.predict(np.array([img]))
        pred_bbox = np.concatenate([
            np.reshape(pred_mbbox, (-1, 5 + params.class_num)),
            np.reshape(pred_lbbox, (-1, 5 + params.class_num))
        ], axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
        bboxes = nms(bboxes, 0.3, method='nms')
        draw_boxes(params, org_img, bboxes)

        return bboxes
    return run_result

def pb_loader(params):
    import tensorflow as tf

    graph = tf.Graph() 
    pb_file = params.pretrain_model  # "./test/test.pb"
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
        draw_boxes(params, org_img, bboxes)
        return bboxes


    return run_result

def tflite_loader(params):
    print("tf - lloader")
    interpreter = tf.lite.Interpreter(model_path=params.pretrain_model)
    interpreter.allocate_tensors()

    # 获取输入和输出张量。
    input_details = interpreter.get_input_details()
    [merge_branch] = interpreter.get_output_details()
    print(input_details)

    @tcost
    def run_result(org_img, input_size, params):
        original_image_size = org_img.shape[:2]
        img = image_preporcess(np.copy(org_img), [input_size, input_size], canny=params.canny)

        input_data = [img.astype(np.float32)]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        bboxes = interpreter.get_tensor(merge_branch["index"])

        pred_bbox = np.reshape(bboxes, (-1, 5 + params.class_num))

        bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.2)
        bboxes = nms(bboxes, 0.3, method='nms')
        draw_boxes(params, org_img, bboxes)
        return bboxes
    return run_result



def model_loader(params):
    if params.pretrain_model.find("pb") != -1:
        return pb_loader(params)
    elif params.pretrain_model.find("tflite") != -1:
        return tflite_loader(params)
    else:
        return checkpoint_loader(params) 

def run_test(params):
    proc = model_loader(params)

    org_img = img = cv2.imread("./data/test/344450.jpg")
    input_size = params.test_input
    result = proc(org_img, input_size, params)
    print("!!!", result)
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

    print(model.outputs, model.inputs)
    # unfriendly ops: tf.newaxis, dims more than 4
    mid, lge = model.outputs
    if params.tflite:
        mid = tf.reshape(mid, (tf.shape(mid)[0], -1, tf.shape(mid)[-1],)) 
        lge = tf.reshape(lge, (tf.shape(lge)[0], -1, tf.shape(lge)[-1],)) 
        merge_branch = tf.concat([mid, lge], axis=1)
        
        model.inputs[0].set_shape([1, params.test_input, params.test_input, 3])
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, model.inputs, [merge_branch])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
        tflite_model = converter.convert()
        open("gesture.tflite", "wb").write(tflite_model)
        return

    tf.compat.v1.saved_model.simple_save(K.get_session(),
        outdir,
        inputs={"input": model.inputs[0]},
        outputs={"output0": mid, "output1": lge})

    freeze_graph.freeze_graph(None,
        None,
        None,
        None,
        mid.op.name + "," + lge.op.name,
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
        freezon_graph(params)
        #print(pb_loader(params))
    elif params.mode == "batch":
        run_batch(params)
    else:
        video(params)
