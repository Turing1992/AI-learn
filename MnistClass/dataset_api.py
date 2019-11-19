#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : dataset_api.py
# @Author: ruixi L
# @Date  : 2019/11/15

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt
tf.set_random_seed(777) #设置随机种子
#定义数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#迭代训练
training_epochs = 15
batch_size = 100

#dataset=tf.data.Dataset.from_tensor_slices(mnist.train.next_batch(55000))
dataset=tf.data.Dataset.from_tensor_slices(mnist.train.next_batch(55000)).repeat(training_epochs)
dataset=dataset.shuffle(1000)
dataset=dataset.batch(batch_size)
iterator1=dataset.make_initializable_iterator()
#element=iterator1.get_next()
#x1=element[0]
#y1=element[1]
X,Y=iterator1.get_next()
#sess = tf.Session()
#sess.run(iterator1.initializer)
#while(True):
    #print(sess.run(element))
    #print(sess.run(x1))
    #print(sess.run(y1))
    #print('/n')


nb_classes = 10
#定义占位符
#

#X = tf.placeholder("float", shape=[None, 784])
#Y = tf.placeholder(tf.float32, [None, nb_classes])
#权重和偏置
W = tf.Variable(tf.random_normal([784, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
#预测模型
# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
logits = tf.matmul(X, W) + b
#hypothesis = logits
hypothesis=tf.nn.softmax(logits)
#代价或损失函数
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#梯度下降优化器
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#准确率计算
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #全局变量初始化

sess.run(iterator1.initializer)
'''
avg_cost = 0
total_batch = int(mnist.train.num_examples / batch_size)
epoch=0
while True:
    try:
        c, _ = sess.run([cost, train])
        avg_cost += c / total_batch
      # 显示损失值收敛情况
        print(epoch, avg_cost)
        epoch+=1
    except tf.errors.OutOfRangeError:
        break
'''

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        c, _ = sess.run([cost, train])
        avg_cost += c / total_batch
    # 显示损失值收敛情况
    print(epoch, avg_cost)
print("Accuracy: ", sess.run(accuracy, feed_dict={X: mnist.test.images[:5000], Y: mnist.test.labels[:5000]}))