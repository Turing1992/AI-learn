import tensorflow as tf
import layer as layer


class Mymodel(object):
    def __init__(self, class_num):
        self.class_num = class_num
        self.X = tf.placeholder(tf.float32, [None, 784], name='x')
        # self.Y = tf.placeholder(tf.float32, [None, 10], name='y')

    def forward(self, input_images):
        input_images = tf.reshape(input_images, [-1, 28, 28, 1])
        with tf.variable_scope('conv2d_1'):
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32]), name='w1')  # 卷积核3x3，输入通道1，输出通道32(卷积核个数)
            L1 = layer.conv2d(input_images, W1, strides=1)
            L1 = layer.max_pool(L1, ksize=2, strides=2)


        with tf.variable_scope('conv2d_2'):
            # 第2层卷积，输入图片数据(?, 14, 14, 32)
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01), name='w2')  # 卷积核3x3，输入通道32，输出通道64
            L2 = layer.conv2d(L1, W2, strides=1)
            L2 = layer.max_pool(L2, ksize=2, strides=2)


        with tf.variable_scope('fc'):
            logits = layer.fully_connected(L2, 10)
        return logits

    def get_loss(self, input, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=input), name='loss')
        return loss

    # def get_accuracy(self, input, labels):
    #     hypothesis = self.forward(input)
    #     correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(labels, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    #     return accuracy
