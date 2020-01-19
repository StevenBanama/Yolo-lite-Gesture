import os
import cv2
import numpy as np
import tensorflow as tf
from utils.utils import image_preporcess, postprocess_boxes, nms, draw_bbox, build_params, config_gpu
from utils.dataset import Dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback

tf.compat.v1.disable_eager_execution()
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, UpSampling2D, MaxPooling2D, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LeakyReLU as LReLU
import tensorflow_model_optimization as tfmot


def block_conv(input, kernel_shape, name, padding="same", strides=(2, 2), activation=None, pooling="max", bn=False):
    conv = tf.keras.layers.Conv2D(kernel_shape[-1],
            tuple(kernel_shape[:2]), padding="same", activation=activation,
            # kernel_regularizer=tf.keras.regularizers.l1(0.01),
            # activity_regularizer=tf.keras.regularizers.l2(0.01),
            name=name)(input)
    if bn:
        conv = tf.keras.layers.BatchNormalization()(conv)
    if pooling == "max":
        conv = MaxPooling2D(pool_size=(2, 2), strides=strides, padding='same')(conv)
    return conv

def bbox_giou(boxes1, boxes2):

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou

def bbox_iou(boxes1, boxes2):

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = 1.0 * inter_area / union_area

    return iou


def decode(conv_output, anchors, stride, class_num, name):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
           contains (x, y, w, h, score, probability)
    """
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    anchor_per_scale = len(anchors)

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + class_num))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.nn.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1, name=name)

def focal_loss(target, actual, alpha=1, gamma=2):
    return alpha * tf.pow(tf.abs(target - actual), gamma)

def region_decode(conv1, conv2, class_num, stride, anchor, name):
    branch = tf.concat([conv1, conv2], axis=-1)
    raw_pred = Conv2D(3 * (class_num + 5), (1, 1), activation=None, name=name + "_raw")(branch)  # activation
    pred = decode(raw_pred, anchor, stride=stride, class_num=class_num, name=name)
    return raw_pred, pred

def loss_layer(conv, anchors, stride, class_num, iou_loss_thresh=0.5, max_bbox_per_scale=150):
    conv_shape = tf.shape(conv)
    batch_size, output_size = conv_shape[0], conv_shape[1]
    input_size = stride * output_size

    input_size = tf.cast(input_size, tf.float32)
    anchor_per_scale = len(anchors)

    conv = tf.reshape(conv, (batch_size, output_size, output_size, anchor_per_scale, 5 + class_num))
    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]


    #@tf.function
    def gen_loss(label, pred):
        '''
            label[b, x, y, scale, 5 + cn]
        '''

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]

        bboxes = tf.reshape(label[..., 0:4], (batch_size, -1, 4))  # change to n:4


        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)

        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(respond_bbox * iou, axis=-1), axis=-1)
        print(iou, max_iou)


        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < iou_loss_thresh, tf.float32 )

        conf_focal = focal_loss(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )
        pos_prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
        neg_prob_loss = 0  # 0.1 * respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
        prob_loss = pos_prob_loss + neg_prob_loss  # add bad case so remove respond_bbox
        print(prob_loss)

        b_giou_loss = tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4])
        b_conf_loss = tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4])
        b_prob_loss = tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4])
        giou_loss = tf.reduce_mean(b_giou_loss)
        conf_loss = tf.reduce_mean(b_conf_loss)
        prob_loss = 0. if class_num <= 1 else tf.reduce_mean(b_prob_loss)

        total_loss = giou_loss + conf_loss + prob_loss
        return total_loss
    return gen_loss 

def lite_backbone_net(input):
    block1 = block_conv(input, [3, 3, 3, 16], name="block1", bn=True) 
    block2 = block_conv(block1, [3, 3, 16, 32], activation=LReLU(), name="block2", bn=True)
    block3 = block_conv(block2, [3, 3, 32, 64], activation=LReLU(), name="block3", bn=True)
    block4 = block_conv(block3, [3, 3, 64, 128], activation=LReLU(), name="block4", bn=True) 
    block5 = block_conv(block4, [3, 3, 128, 128], activation=LReLU(), name="block5", bn=True)
    block6 = block_conv(block5, [3, 3, 128, 256], pooling=None, name="block6", bn=True)
    block7 = block_conv(block6, [1, 1, 256, 128], pooling=None, name="block7", bn=True)
    backbone = Model([input], block7)
    backbone.summary() 
    return backbone

def get_callbacks(params):

    return [
        ModelCheckpoint(params.save_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min'),
        TensorBoard(log_dir=params.log_dir, write_images=True, update_freq='epoch'),
        tfmot.sparsity.keras.PruningSummaries("./log")
    ]

def build_model(params):
    input = Input(shape=[None, None, 3])

    checkpoint_dir = os.path.dirname(params.save_path)

    backbone = lite_backbone_net(input)

    with tf.name_scope('branch'):
        conv4 = backbone.get_layer("max_pooling2d_3").output
        conv5 = backbone.get_layer("max_pooling2d_4").output
        conv7 = backbone.get_layer("block7").output

        mid_raw, mid_pred = region_decode(conv4, UpSampling2D(2)(conv5), params.class_num, params.strides[0], params.anchors[0], "mid")
        lge_raw, lge_pred = region_decode(conv5, conv7, params.class_num, params.strides[1], params.anchors[1], "large")

    models = Model([input], [mid_pred, lge_pred])

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0),
        'block_size': (1, 1),
        'block_pooling_type': 'MAX'
    }


    if params.train:
        adam = Adam(lr=params.lr)
        models.compile(
            optimizer=adam,
            loss=[
                loss_layer(mid_raw, params.anchors[0], params.strides[0], params.class_num, iou_loss_thresh=params.iou_thres),
                loss_layer(lge_raw, params.anchors[1], params.strides[1], params.class_num, iou_loss_thresh=params.iou_thres)
            ],
        )
    if params.restore:
        models.load_weights(params.pretrain_model)
        # models = load_model(params.pretrain_model)
        print("!!!!!!!")
    #models = tfmot.sparsity.keras.prune_low_magnitude(models, **pruning_params)
    return models

def build_net(params):
    dataset = Dataset("train", params, pworker=1)
    testset = Dataset("test", params, pworker=1)

    models = build_model(params)
    print(models.summary())
    
    models.fit_generator(dataset.gen_iter(),
        steps_per_epoch=dataset.num_batchs, epochs=params.epoch,
        validation_data=testset.gen_iter(),
        validation_steps=testset.num_batchs,
        callbacks=get_callbacks(params)
    )


if __name__ == "__main__":
    config_gpu()
    params = build_params()
    build_net(params)
