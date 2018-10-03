import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.slim as slim

HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIMES_STEPS = 10
TRAINING_STEPS = 500
BATCH_SIZE = 512

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01

def generate_data(seq):
	X = []
	y = []
	for i in range(len(seq) - TIMES_STEPS):
		X.append([seq[i:i+TIMES_STEPS]])
		y.append([seq[i+TIMES_STEPS]])
	return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def lstm_model(X, y, is_training):

	cell = tf.nn.rnn_cell.MultiRNNCell(
		[tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) 
		for _ in range(NUM_LAYERS)])

	outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
	output = outputs[:, -1, :]

	predictions = slim.fully_connected(output, 1, activation_fn=None)

	if not is_training:
		return predictions, None, None

	loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

	train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

	return predictions, loss, train

def train(sess, train_X, train_y):

	dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
	dataset = dataset.shuffle(1000).batch(BATCH_SIZE).repeat()
	X, y = dataset.make_one_shot_iterator().get_next()

	with tf.variable_scope('model'):
		predictions, loss, train = lstm_model(X, y, True)

	sess.run(tf.global_variables_initializer())

	for i in range(TRAINING_STEPS):
		_, l = sess.run([train, loss])
		if i % 100 == 0:
			print('train step: %d , loss: %g' % (i, l))

def eval(sess, test_X, test_y):
	dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
	dataset = dataset.batch(1)
	X, y = dataset.make_one_shot_iterator().get_next()

	with tf.variable_scope('model', reuse=True):
		prediction, _, _ = lstm_model(X, y, False)

	predictions = []
	labels = []
	for i in range(TESTING_EXAMPLES):
		p, l = sess.run([prediction, y])
		predictions.append(p)
		labels.append(l)

	predictions = np.array(predictions).squeeze()
	labels = np.array(labels).squeeze()

	rmse = np.sqrt(((predictions-labels) ** 2).mean(axis=0))
	print('Root Square Error is: %f' % rmse)

	plt.figure()
	plt.plot(predictions, label='predicitons')
	plt.plot(labels, label='real_sin')
	plt.legend()
	plt.show()


test_start = (TRAINING_EXAMPLES + TIMES_STEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMES_STEPS) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(
	0, test_start, TRAINING_EXAMPLES+TIMES_STEPS, dtype=np.float32)))

test_X, test_y = generate_data(np.sin(np.linspace(
	test_start, test_end, TESTING_EXAMPLES+TIMES_STEPS, dtype=np.float32)))

with tf.Session() as sess:
	train(sess, train_X, train_y)
	eval(sess, test_X, test_y)