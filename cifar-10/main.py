import input_data
import numpy as np
import tensorflow as tf 
np.set_printoptions(threshold=np.inf)

def convert_to_one_hot(array, C):
    return np.eye(C)[array.reshape(-1)]

def uniformization(array):
	return array / 255.

def weight_varialbe(shape,stddev):
	initial = tf.truncated_normal(shape=shape,stddev=stddev)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(shape=shape,value=0.1)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')

def avg_pool_2x2(x):
	return tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')

#读入数据
cifar10 = input_data.read_data_sets('CIFAR-10_data/')
#归一化
cifar10.train.images, cifar10.test.images = uniformization(cifar10.train.images), uniformization(cifar10.test.images)
#OneHot编码
cifar10.train.labels, cifar10.test.labels = convert_to_one_hot(cifar10.train.labels,10), convert_to_one_hot(cifar10.test.labels,10)


x = tf.placeholder(dtype=tf.float32,shape=[None,3072])
x_image = tf.reshape(x,[-1,32,32,3])
#第一层卷积
W_conv1 = weight_varialbe([3,3,3,32],1e-4)
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#第二层卷积
W_conv2 = weight_varialbe([4,4,32,32],1e-3)
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = avg_pool_2x2(h_conv2)
#第三层卷积
W_conv3 = weight_varialbe([5,5,32,64],1e-2)
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
h_pool3 = avg_pool_2x2(h_conv3)
#全连接层
W_fc1 = weight_varialbe([32*32*64,1024],0.1)
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool3,[-1,32*32*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1) + b_fc1)
#Dropout
keep_prob = tf.placeholder(dtype=tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
#输出层
W_fc2 = weight_varialbe([1024,10],0.1)
b_fc2 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
y_ = tf.placeholder(dtype=tf.float32,shape=[None,10])

cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-8,1)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#评估模型
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#初始化
init = tf.initialize_all_variables()
sess = tf.Session()

print(h_pool3.shape)
for i in range(30000):
	batch_xs, batch_ys = cifar10.train.next_batch(200)
	sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
	if i % 100 == 0:
		test_batch_xs, test_batch_ys = cifar10.test.next_batch(1000)
		test_accuracy, loss = sess.run([accuracy, cross_entropy] ,feed_dict={x:test_batch_xs,y_:test_batch_ys,keep_prob:1.0})
		print("step %d, test accuracy %g, loss %g" % (i, test_accuracy, loss))
	