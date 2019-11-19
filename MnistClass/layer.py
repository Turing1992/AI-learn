import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import layers


# tf.layers.conv2d(inputs=,filters=,kernel_size=,strides=(1,1))
# tf.nn.conv2d(input=, filter=, strides=, padding=)
# slim.conv2d(activation_fn=)
# tf.nn.max_pool(value=, ksize=, strides=, padding=)
# tf.nn.dropout(x=, keep_prob=)
# layers.fully_connected(inputs=, num_outputs=, activation_fn=None,)


def conv2d(input, filter, strides=1, padding='SAME', is_activate=True):
    output = tf.nn.conv2d(input, filter, strides=[1, strides, strides, 1], padding=padding)
    if is_activate:
        return tf.nn.relu(output)
    else:
        return output


def max_pool(value, ksize=2, strides=2, padding='SAME', is_dropout=False, keep_prob=0.9):
    output = tf.nn.max_pool(value, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding=padding)
    if is_dropout:
        output = tf.nn.dropout(output, keep_prob)
        return output
    else:
        return output


def fully_connected(input, num_outputs):
    if len(input.get_shape()) > 2:
        dim = input.get_shape()[1].value * input.get_shape()[2].value * input.get_shape()[3].value
        L2_flag = tf.reshape(input, [-1, dim])
        W3 = tf.Variable(tf.random_normal([dim, num_outputs], stddev=0.01), name='w3')
        b = tf.Variable(tf.random_normal([num_outputs]), name='b')
        output = tf.add(tf.matmul(L2_flag, W3), b, name='logits')
        return output
    else:
        dim = input.get_shape()[1].value
        W3 = tf.Variable(tf.random_normal([dim, num_outputs], stddev=0.01), name='w3')
        b = tf.Variable(tf.random_normal([num_outputs]), name='b')
        output = tf.add(tf.matmul(input, W3), b, name='logits')
        return output


def loss(pre_value, tru_value):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tru_value, logits=pre_value))
    return loss
