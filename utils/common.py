import tensorflow.compat.v1 as tf

def Conv2D(inputs, filters, name, strides=[1, 1, 1, 1], padding="SAME", pooling=None, activation="leaky", trainable=True, bn=False):
    with tf.variable_scope(name):
        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True, shape=filters, initializer=tf.random_normal_initializer(stddev=0.01))

        output = tf.nn.conv2d(inputs, weight, strides=strides, padding=padding)
        if pooling == "max":
            output = tf.nn.max_pool(output, ksize=[1], strides=[1, 2, 2, 1], padding="SAME") 
        elif pooling == "avg":
            output = tf.nn.avg_pool(output, ksize=[1], strides=[1, 2, 2, 1], padding="SAME") 
 
        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=trainable)

        if activation == "leaky":
            output = tf.nn.leaky_relu(output) 
        elif activation == "relu":
            output = tf.nn.leaky_relu(output)
    return output

def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output


