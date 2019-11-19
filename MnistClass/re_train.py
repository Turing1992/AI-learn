#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : re_train.py
# @Author: ruixi L
# @Date  : 2019/11/15

# 导入依赖包（5分）
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# 输入数据（5分）
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

y_input=tf.placeholder(tf.int32,shape=[None,10],name='y')

saver=tf.train.import_meta_graph('./checkpoint/mymodel.meta')
graph=tf.get_default_graph()
x_input=graph.get_tensor_by_name('x:0')
logits=graph.get_tensor_by_name('fc/logits:0')

prediction = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_input, axis=1)), dtype=tf.float32))
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('./checkpoint'))

    # a=sess.run(accuracy,feed_dict={x_input:mnist.test.images[:5000],y_input:mnist.test.labels[:5000]})
    # print(a)

    cost_re=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_input))
    optimizer=tf.train.AdamOptimizer(0.01).minimize(cost_re)

    g_var=[]
    train_var=tf.trainable_variables()
    train_nonevar=tf.global_variables()
    for i in range(len(train_nonevar)):
        if train_nonevar[i] in train_var:
            continue
        else:
            g_var.append(train_nonevar[i])
    sess.run(tf.variables_initializer(var_list=g_var))

    for step in range(5):
        avg_cost=0
        total_batch=int(mnist.train.num_examples/100)
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(100)
            cost_val,_=sess.run([cost_re,optimizer],feed_dict={x_input:batch_xs,y_input:batch_ys})
            avg_cost+=cost_val/total_batch
        print(step+1,avg_cost)
    print('accuracy',sess.run(accuracy,feed_dict={x_input:mnist.test.images[:5000],y_input:mnist.test.labels[:5000]}))
