import input_data
import tensorflow as tf 
import numpy as np


def weight_variable(shape):
	W = tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1), name='weight')
	return W


def bais_variable(shape):
	b = tf.get_variable(shape=shape, initializer=tf.constant_initializer(value=0.1), name='bias')
	return b


def conv2d(x, W, name):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name='conv')


def max_pool_2x2(x, name):
	with tf.variable_scope(name):
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool')


def conv_layer(x, kernel_shape, name):
	with tf.variable_scope(name):
		W = weight_variable(kernel_shape)
		b = bais_variable(kernel_shape[3])
		return tf.nn.relu(conv2d(x, W, name) + b, name='relu')


def fc_layer(x, shape, name, softmax=False):
	with tf.variable_scope(name):
		W = weight_variable(shape)
		b = bais_variable(shape[1])
		if not softmax:
			return tf.nn.relu(tf.matmul(x, W) + b, name='relu')
		else:
			return tf.nn.softmax(tf.matmul(x, W) + b, name='softmax')


INPUT_SIZE = 784
IMAGE_WIDTH = 28
IMAGE_HETGHT = 28
IMAGE_CHANNEL = 1
CLASS_NUM = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
KEEP_PROB = 0.5


with tf.variable_scope('input'):
	x = tf.placeholder(dtype=tf.float32, shape=[None,INPUT_SIZE], name='data')
	x_image = tf.transpose(tf.reshape(x, [-1,IMAGE_CHANNEL,IMAGE_WIDTH,IMAGE_HETGHT]), [0,3,2,1], name='image')
	y_ = tf.placeholder(dtype=tf.float32, shape=[None,CLASS_NUM], name='label')
	keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')


with tf.variable_scope('layer'):
	net = conv_layer(x_image, [5,5,x_image.shape[3],32], name='conv1')
	net = max_pool_2x2(net, name='pool1')
	net = conv_layer(net, [5,5,net.shape[3],64], name='conv2')
	net = max_pool_2x2(net, name='pool2')
	net = tf.reshape(net, [-1,net.shape[1]*net.shape[2]*net.shape[3]], name='flatten')
	net = fc_layer(net, [net.shape[1],1024], name='fc1')
	net = tf.nn.dropout(net, keep_prob, name='drop1')


with tf.variable_scope('output'):
	y = fc_layer(net, [net.shape[1],CLASS_NUM], name='output', softmax=True)


with tf.variable_scope('loss'):
	cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-9,1)), name='loss')


with tf.variable_scope('train'):
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)


with tf.variable_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')


writer = tf.summary.FileWriter('log/', tf.get_default_graph())
#writer.close()


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		batch_xs,batch_ys = mnist.train.next_batch(BATCH_SIZE)
		if i % 10 == 0:
			run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys,keep_prob:KEEP_PROB},
				options=run_option, run_metadata=run_metadata)
			writer.add_run_metadata(run_metadata, 'step%d' % i)
		else:
			sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys,keep_prob:KEEP_PROB})
		if i % 100 == 0:
			train_accuracy = sess.run(accuracy, feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1.0},
				options=run_option, run_metadata=run_metadata)
			print("step %d, training accuracy %g" % (i,train_accuracy))


	print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))