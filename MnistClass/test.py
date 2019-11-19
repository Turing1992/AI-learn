#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: ruixi L
# @Date  : 2019/11/14

import argparse
import tensorflow as tf
import numpy as np
from cnn_model import CNNMnist
import cv2

parser=argparse.ArgumentParser(description='choice images in mnist')
parser.add_argument("input_image", type=str,default='./MNIST_data',
                    help="The path of the input image.")
# parser.add_argument("--new_size", nargs='*', type=int, default=[-1,28,28,1],)

args = parser.parse_args()
img=cv2.imread(args.input_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.

with tf.Session() as sess :
    input_image=tf.placeholder('float',shape=[None,784],name='x')
    model=CNNMnist()
    with tf.variable_scope('test'):
        logits=model.foward(input_image)
    sess.run(tf.global_variables_initializer())
    logits=sess.run(logits,feed_dict={input_image:img})
    print('预测值',logits)
