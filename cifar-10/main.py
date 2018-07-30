import input_data
import numpy as np
import tensorflow as tf 
np.set_printoptions(threshold=np.inf)

def convert_to_one_hot(array, C):
    return np.eye(C)[array.reshape(-1)]

def uniformization(array):
	return array / 255.

scope = 0

def weight_varialbe(shape):
	global scope
	scope += 1
	return tf.get_variable(str(scope),shape=shape,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d()) 

def bias_variable(shape,value):
	initial = tf.constant(shape=shape,value=value)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def conv_layer_3x3(x,input,output):
	W_conv = weight_varialbe([3,3,input,output])
	b_conv = bias_variable([output],0.0)
	return tf.nn.relu(conv2d(x,W_conv) + b_conv)

def fc_layer(x,input,output):
	W_fc = weight_varialbe([input,output])
	b_fc = bias_variable([output],0.1)
	return tf.nn.relu(tf.matmul(x,W_fc) + b_fc)

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



cifar10 = input_data.read_data_sets('CIFAR-10_data/')

cifar10.train.images, cifar10.test.images = uniformization(cifar10.train.images), uniformization(cifar10.test.images)

cifar10.train.labels, cifar10.test.labels = convert_to_one_hot(cifar10.train.labels,10), convert_to_one_hot(cifar10.test.labels,10)


x = tf.placeholder(dtype=tf.float32,shape=[None,3072])
x_image = tf.reshape(x,[-1,32,32,3])
h_conv1 = conv_layer_3x3(x_image,3,64)
h_conv2 = conv_layer_3x3(h_conv1,64,64)
h_pool1 = max_pool_2x2(h_conv2)
h_conv3 = conv_layer_3x3(h_pool1,64,128)
h_conv4 = conv_layer_3x3(h_conv3,128,128)
h_pool2 = max_pool_2x2(h_conv4)
h_conv5 = conv_layer_3x3(h_pool2,128,256)
h_conv6 = conv_layer_3x3(h_conv5,256,256)
h_conv7 = conv_layer_3x3(h_conv6,256,256)
h_pool3 = max_pool_2x2(h_conv7)
h_conv8 = conv_layer_3x3(h_pool3,256,512)
h_conv9 = conv_layer_3x3(h_conv8,512,512)
h_conv10 = conv_layer_3x3(h_conv9,512,512)
h_pool4 = max_pool_2x2(h_conv10)
h_conv11 = conv_layer_3x3(h_pool4,512,512)
h_conv12 = conv_layer_3x3(h_conv11,512,512)
h_conv13 = conv_layer_3x3(h_conv12,512,512)
h_pool5 = max_pool_2x2(h_conv13)
h_pool5_flat = tf.reshape(h_pool5,[-1,512])
h_fc1 = fc_layer(h_pool5_flat,512,4096)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
h_fc2 = fc_layer(h_fc1_drop,4096,4096)
h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)
h_fc3 = fc_layer(h_fc2_drop,4096,1000)
W_fc4 = weight_varialbe([1000,10])
b_fc4 = bias_variable([10],0.1)
y = tf.nn.softmax(tf.matmul(h_fc3_drop,W_fc4) + b_fc4,name='predictions')
y_ = tf.placeholder(dtype=tf.float32,shape=[None,10],name='labels')

global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(1e-2,global_step,decay_steps=13600,decay_rate=0.1,staircase=True) 
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-8,1)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(100000):
	batch_xs, batch_ys = cifar10.train.next_batch(128)
	sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
	if i % 100 == 0:
		test_batch_xs, test_batch_ys = cifar10.test.next_batch(100)
		test_accuracy, loss = sess.run([accuracy, cross_entropy] ,feed_dict={x:test_batch_xs,y_:test_batch_ys,keep_prob:1.0})
		print("step %d, test accuracy %g, loss %g" % (i, test_accuracy, loss))
