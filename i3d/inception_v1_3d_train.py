import inception_v1_3d
import ckpt_util
import input_data
import numpy as np 
import os
import tensorflow.contrib.slim as slim
import tensorflow as tf 


'''
建立计算图
从ckpt文件中恢复参数
创建数据集
'''
x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 224, 224, 3])
keep_prob = tf.placeholder(dtype=tf.float32)
logits = inception_v1_3d.inception_v1_3d(x, keep_prob, 101)

y = tf.nn.softmax(logits)
y_ = tf.placeholder(dtype=tf.float32,shape=[None, 101])

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y, 1e-7, 1)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt_util.restore(sess)
one_element = input_data.read_data(sess)
for i in range(3000):
	element = sess.run(one_element)
	datas = (tf.cast(element[0], tf.float32) / 128. -1).eval(session=sess)
	labels = input_data.one_hot([b.decode() for b in element[1].tolist()])
	sess.run(train_step, feed_dict={x:datas, y_:labels, keep_prob:0.6})
	if i % 100 == 0:
		element = sess.run(one_element)
		datas = tf.cast(element[0], tf.float32) / 128. -1
		labels = input_data.one_hot([b.decode() for b in element[1].tolist()])
		test_accuracy, loss = sess.run([accuracy, cross_entropy] ,feed_dict={x:datas, y_:labels, keep_prob:1})
		print("step %d, test accuracy %g, loss %g" % (i, test_accuracy, loss))