import math
import tensorflow as tf


def linear_op(X, n_units):
    input_dim = X.get_shape().as_list()[-1]
    w = tf.Variable(tf.truncated_normal([input_dim, n_units], stddev=1e-1), name="weights")
    b = tf.Variable(tf.zeros([n_units]), name="biases")
    return tf.add(tf.matmul(X, w), b)


def fully_connected(X, n_outputs):
    return tf.tanh(linear_op(X, n_outputs))


def inference(X, n_units, output_dim):
    with tf.name_scope("hidden1"):
        hidden1 = fully_connected(X, n_units)
    with tf.name_scope("hidden2"):
        hidden2 = fully_connected(hidden1, n_units)
    with tf.name_scope("output"):
        # return linear_op(hidden2, output_dim)
        return fully_connected(hidden2, output_dim)


def loss(y, gt):
    with tf.name_scope("loss"):
        # loss = tf.reduce_sum((y - gt)**2)
        loss = tf.nn.l2_loss(y-gt)
        return tf.Print(loss, [tf.reduce_min(y), tf.reduce_max(y), tf.reduce_min(gt), tf.reduce_max(gt)])


def training_error(loss, gt):
    with tf.name_scope("error"):
        return (loss*2) / tf.reduce_sum(gt**2)


def training(loss, lr):
    with tf.name_scope("train"):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        return optimizer.minimize(loss, global_step=global_step)
