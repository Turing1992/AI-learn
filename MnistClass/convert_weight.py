#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : convert_weight.py
# @Author: ruixi L
# @Date  : 2019/11/13


import tensorflow as tf
from model import Mymodel
from utils import load_weight

weight_path=r'./model.pb'
inputs=tf.placeholder(tf.float32,shape=[None,784],name='x')
model=Mymodel(class_num=10)
logits=model.forward(inputs)
name_list=tf.trainable_variables()
with tf.Session() as sess:
    load_op=load_weight(name_list,weight_path)
    for i in range(len(load_op)):
        sess.run(tf.assign(name_list[i],load_op[i]))
    tf.train.Saver().save(sess,'./checkpoint/mymodel')
    print('权重转换完成')

