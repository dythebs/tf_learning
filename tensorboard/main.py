import input_data
import tensorflow as tf 
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import os


def weight_variable(shape):
	W = tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1), name='weight')
	#将张量加入统计
	tf.summary.histogram('summary_weight', W)
	return W


def bais_variable(shape):
	b = tf.get_variable(shape=shape, initializer=tf.constant_initializer(value=0.1), name='bias')
	tf.summary.histogram('summary_bias', b)
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
		ac = tf.nn.relu(conv2d(x, W, name) + b, name='relu')
		tf.summary.histogram('summary_ac', ac)
		return ac


def fc_layer(x, shape, name, softmax=False):
	with tf.variable_scope(name):
		W = weight_variable(shape)
		b = bais_variable(shape[1])
		if not softmax:
			ac = tf.nn.relu(tf.matmul(x, W) + b, name='relu')
		else:
			ac = tf.nn.softmax(tf.matmul(x, W) + b, name='softmax')
		tf.summary.histogram('summary_ac', ac)
		return ac


INPUT_SIZE = 784
IMAGE_WIDTH = 28
IMAGE_HETGHT = 28
IMAGE_CHANNEL = 1
CLASS_NUM = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
KEEP_PROB = 0.5
LOG_DIR = 'log/'
SPRITE_FILE = 'mnist_sprite.jpg'
META_FILE = 'mnist_meta.tsv'


with tf.variable_scope('input'):
	x = tf.placeholder(dtype=tf.float32, shape=[None,INPUT_SIZE], name='data')
	x_image = tf.reshape(x, [-1,IMAGE_HETGHT,IMAGE_WIDTH,IMAGE_CHANNEL], name='image')
	#将图像加入统计
	tf.summary.image('summary_image', x_image, 4)
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
	#将标量加入统计
	tf.summary.scalar('summary_loss', cross_entropy)


with tf.variable_scope('train'):
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)


with tf.variable_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')
	tf.summary.scalar('summary_accuracy', accuracy)


#合并所有的统计
merged = tf.summary.merge_all()
#写入计算图
writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		batch_xs,batch_ys = mnist.train.next_batch(BATCH_SIZE)
		if i % 10 == 0:
			#统计运行的时间和内存
			run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			summary, _ = sess.run([merged, train_step], feed_dict={x:batch_xs,y_:batch_ys,keep_prob:KEEP_PROB},
				options=run_option, run_metadata=run_metadata)
			writer.add_run_metadata(run_metadata, 'step%d' % i)
			#将merged的统计加入log
			writer.add_summary(summary, i)
		else:
			sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys,keep_prob:KEEP_PROB})
		if i % 100 == 0:
			train_accuracy = sess.run(accuracy, feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1.0})
			print("step %d, training accuracy %g" % (i, train_accuracy))

	final_result = sess.run(y, feed_dict={x:mnist.test.images,keep_prob:1.0})


#可视化结果
#使用结果创建Tensor
y = tf.Variable(final_result, name='final_result')
writer = tf.summary.FileWriter(LOG_DIR)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()

embedding.tensor_name = y.name
embedding.metadata_path = META_FILE
embedding.sprite.image_path = SPRITE_FILE
embedding.sprite.single_image_dim.extend([28,28])

projector.visualize_embeddings(writer, config)
writer.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
Saver = tf.train.Saver()
Saver.save(sess, os.path.join(LOG_DIR, 'model'))