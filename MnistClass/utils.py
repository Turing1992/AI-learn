#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: ruixi L
# @Date  : 2019/11/13

import tensorflow as tf

from tensorflow.python.platform import gfile

def load_weight(var_list,weight_path):
    with tf.Session() as sess:
        with gfile.FastGFile(weight_path,'rb') as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            name_list=[j.name for j in var_list]
            values=tf.import_graph_def(graph_def,return_elements=name_list,name='')
    return values