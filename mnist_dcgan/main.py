import input_data
import tensorflow.contrib.slim as slim
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
BETA1 = 0.4
NOISE_SIZE = 100
tf.reset_default_graph()


def generator(z):
	with slim.arg_scope([slim.fully_connected], 
		normalizer_fn=slim.batch_norm, 
		activation_fn=tf.nn.relu
		):
		net = slim.fully_connected(z, 1024)
		net = slim.fully_connected(net, 128*7*7)
		net = tf.reshape(net, [-1, 7, 7, 128])
	with slim.arg_scope([slim.conv2d_transpose], 
		normalizer_fn=slim.batch_norm, 
		kernel_size=5, stride=2, padding='SAME', 
		activation_fn=tf.nn.relu
		):
		net = slim.conv2d_transpose(net, 128)
		net = slim.conv2d_transpose(net, 1, activation_fn=tf.nn.tanh, normalizer_fn=None)
		return net


def discriminator(x):
	with slim.arg_scope([slim.conv2d], 
		activation_fn=tf.nn.leaky_relu,
		kernel_size=5,
		normalizer_fn=slim.batch_norm, 
		stride=2
		):
		with slim.arg_scope([slim.fully_connected], 
			activation_fn=tf.nn.leaky_relu
			):
			net = slim.conv2d(x, 64)
			net = slim.conv2d(net, 128)
			net = tf.reshape(net, [-1, 128*7*7])
			net = slim.fully_connected(net, 1024)
			net = slim.fully_connected(net, 1, activation_fn=None)
			return tf.nn.sigmoid(net), net


def get_input_tensor():
	mnist = input_data.read_data_sets("MNIST_data/")

	images = np.concatenate((mnist.train.images, mnist.validation.images, 
		mnist.test.images))
	images = np.reshape(images, [-1, 28, 28, 1])

	dataset = tf.data.Dataset.from_tensor_slices(images)
	dataset = dataset.batch(BATCH_SIZE).shuffle(10000).repeat()
	one_element = dataset.make_one_shot_iterator().get_next()

	return one_element


z = tf.random_uniform([BATCH_SIZE, 100], -1, 1, dtype=tf.float32)
x = get_input_tensor()
with tf.variable_scope('generator'):
	g_out = generator(z)
with tf.variable_scope('discriminator'):
	d_real_out, d_real_logits = discriminator(x)
with tf.variable_scope('discriminator', reuse=True):
	d_fake_out, d_fake_logits = discriminator(g_out)


g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_out)))
d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_out)))
d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_out)))
d_loss = d_real_loss + d_fake_loss

var_list = tf.trainable_variables()
g_var_list = [var for var in var_list if var.name.startswith('generator')]
d_var_list = [var for var in var_list if var.name.startswith('discriminator')]

g_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(g_loss, var_list=g_var_list)
d_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(d_loss, var_list=d_var_list)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(200000):
		sess.run(g_optim)
		sess.run(d_optim)
		sess.run(g_optim)
		if i % 100 == 0:
			image, gl, drl, dfl = sess.run([g_out, g_loss, d_real_loss, d_fake_loss])
			print("step %d, g_loss %g, dr_loss %g, df_loss %g" % (i, gl, drl, dfl))
			image = (image+1)*127.5
			full_images = np.zeros([28*8, 28*8])
			for i_ in range(8):
				for j_ in range(8):
					full_images[i_*28:(i_+1)*28, j_*28:(j_+1)*28] = image[i_*8+j_,:,:,0]
			plt.imshow(full_images, cmap='gray')
			plt.grid(False)
			plt.show()
