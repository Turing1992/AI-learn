#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: ruixi L
# @Date  : 2019/11/13


import tensorflow as tf
from model import Mymodel
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
model = Mymodel(class_num=10)

with tf.Session() as sess:
    dataset = tf.data.Dataset.from_tensor_slices(mnist.train.next_batch(55000)).repeat(15)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(100)
    iterator1 = dataset.make_initializable_iterator()
    X, Y = iterator1.get_next()
    # X = tf.placeholder('float', shape=[None, 784], name='x')
    # Y = tf.placeholder(tf.int32, shape=[None, 10], name='y')

    logits = model.forward(X)
    cost = model.get_loss(logits, Y)

    optimizer=tf.train.AdamOptimizer(0.01).minimize(cost)

    # prediction = tf.argmax(logits, axis=1)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y, axis=1)), dtype=tf.float32))
    sess.run(tf.global_variables_initializer())
    sess.run(iterator1.initializer)
    var_list = tf.trainable_variables()
    var_list = [i.name.split(':')[0] for i in var_list]


    for step in range(15):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / 100)
        for i in range(total_batch):
            # batch_xs,batch_ys=mnist.train.next_batch(model.batch_size)
            cost_val,_=sess.run([cost,optimizer])
            avg_cost+=cost_val/total_batch
        print(step+1,avg_cost)
    # print('accuracy',sess.run(accuracy,feed_dict={X:mnist.test.images[:5000],Y:mnist.test.labels[:5000]}))
    with tf.gfile.FastGFile('model.pb','wb') as f:
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, var_list)
        f.write(constant_graph.SerializeToString())
