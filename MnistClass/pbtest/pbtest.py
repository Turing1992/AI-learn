#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : pbtest.py
# @Author: ruixi L
# @Date  : 2019/11/15

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

#先定义计算图
a=tf.Variable(6.0,name='a')
b=tf.Variable(7.0,name='b')
c=a+b