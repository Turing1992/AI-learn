#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : cnn_model.py
# @Author: ruixi L
# @Date  : 2019/11/13


import tensorflow as tf
import numpy as np

class CNNMnist(object):
    def __init__(self,class_num,learning_rate=0.01,training_epoch=15,batch_size=100,defluat=None):
        self.class_num=class_num
        self.learning_rate=learning_rate
        self.training_epoch=training_epoch
        self.batch_size=batch_size
        self.defluat=defluat

    def foward(self,image_inputs):
        inputs=tf.reshape(image_inputs,[-1,28,28,1])
        with tf.variable_scope('conv1'):
            w1=tf.Variable(tf.random_normal([3,3,1,32]),name='w1')
            net=tf.nn.conv2d(inputs,w1,strides=[1,1,1,1],padding='SAME')
            net=tf.nn.relu(net)
            pool1=tf.nn.max_pool(net,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        with tf.variable_scope('conv2'):
            w2=tf.Variable(tf.random_normal([3,3,32,64]),name='w2')
            net=tf.nn.conv2d(pool1,w2,strides=[1,1,1,1],padding='SAME')
            net=tf.nn.relu(net)
            pool2=tf.nn.max_pool(net,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        with tf.variable_scope('fully_connected'):

            dim = pool2.get_shape()[1].value * pool2.get_shape()[2].value * pool2.get_shape()[3].value
            L2_flag = tf.reshape(pool2, [-1, dim])
            W3 = tf.Variable(tf.random_normal([dim, 10], stddev=0.01), name='w3')
            b = tf.Variable(tf.random_normal([10]), name='b')
            logits = tf.add(tf.matmul(L2_flag, W3), b,name='logits')

        return logits