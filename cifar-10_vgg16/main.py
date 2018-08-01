import input_data
import numpy as np
import tensorflow as tf 

def convert_to_one_hot(array, C):
	return np.eye(C)[array.reshape(-1)]

def uniformization(array):
	return array / 255.

def weight_varialbe(shape,name):
	var = tf.get_variable(name,shape=shape,dtype=tf.float32,initializer=tf.contrib.keras.initializers.he_normal())
	tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(0.001)(var))
	return var

def bias_variable(shape,value):
	initial = tf.constant(shape=shape,value=value)
	return tf.Variable(initial)

def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3, updates_collections=None)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def conv_layer_3x3(x,input,output,name):
	W_conv = weight_varialbe([3,3,input,output],name)
	b_conv = bias_variable([output],0.0)
	return tf.nn.relu(batch_norm(conv2d(x,W_conv) + b_conv))

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def fc_layer(x,input,output):
	W_fc = weight_varialbe([input,output])
	b_fc = bias_variable([output],0.1)
	return tf.nn.relu(tf.matmul(x,W_fc) + b_fc)

#读入数据
cifar10 = input_data.read_data_sets('CIFAR-10_data/')
#归一化
cifar10.train.images, cifar10.test.images = uniformization(cifar10.train.images), uniformization(cifar10.test.images)
#OneHot编码
cifar10.train.labels, cifar10.test.labels = convert_to_one_hot(cifar10.train.labels,10), convert_to_one_hot(cifar10.test.labels,10)
#实现模型
x = tf.placeholder(dtype=tf.float32,shape=[None,3072])
x_image = tf.transpose(tf.reshape(x,[-1,3,32,32]),[0,3,2,1])
keep_prob = tf.placeholder(tf.float32)
h_conv1 = conv_layer_3x3(x_image,3,64,'conv_1')
h_conv2 = conv_layer_3x3(h_conv1,64,64,'conv2')
h_pool1 = max_pool_2x2(h_conv2)
h_pool1 = tf.nn.dropout(h_pool1,keep_prob)
h_conv3 = conv_layer_3x3(h_pool1,64,128,'conv3')
h_conv4 = conv_layer_3x3(h_conv3,128,128,'conv4')
h_pool2 = max_pool_2x2(h_conv4)
h_pool2 = tf.nn.dropout(h_pool2,keep_prob)
h_conv5 = conv_layer_3x3(h_pool2,128,256,'conv5')
h_conv6 = conv_layer_3x3(h_conv5,256,256,'conv6')
h_conv7 = conv_layer_3x3(h_conv6,256,256,'conv7')
h_pool3 = max_pool_2x2(h_conv7)
h_pool3 = tf.nn.dropout(h_pool3,keep_prob)
h_conv8 = conv_layer_3x3(h_pool3,256,512,'conv8')
h_conv9 = conv_layer_3x3(h_conv8,512,512,'conv9')
h_conv10 = conv_layer_3x3(h_conv9,512,512,'conv10')
h_pool4 = max_pool_2x2(h_conv10)
h_pool4 = tf.nn.dropout(h_pool4,keep_prob)
h_conv11 = conv_layer_3x3(h_pool4,512,512,'conv11')
h_conv12 = conv_layer_3x3(h_conv11,512,512,'conv12')
h_conv13 = conv_layer_3x3(h_conv12,512,512,'conv13')
h_pool5 = max_pool_2x2(h_conv13)
h_pool5_flat = tf.reshape(h_pool5,[-1,512])
h_pool5_flat_drop = tf.nn.dropout(h_pool5_flat,keep_prob)
W_fc4 = weight_varialbe([512,10],'fc')
b_fc4 = bias_variable([10],0.1)
y = tf.nn.softmax(tf.matmul(h_pool5_flat_drop,W_fc4) + b_fc4,name='predictions')
y_ = tf.placeholder(dtype=tf.float32,shape=[None,10],name='labels')
#指数衰减学习率
global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(1e-3,global_step,decay_steps=4000,decay_rate=0.1,staircase=True) 
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-9,1)))
#L2正则化
tf.add_to_collection('losses',cross_entropy)
loss = tf.add_n(tf.get_collection('losses'))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#训练模型
for i in range(20000):
	batch_xs, batch_ys = cifar10.train.next_batch(256)
	sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
	if i % 100 == 0:
		test_batch_xs, test_batch_ys = cifar10.test.next_batch(2000)
		test_accuracy, loss = sess.run([accuracy, cross_entropy] ,feed_dict={x:test_batch_xs,y_:test_batch_ys,keep_prob:1.0})
		print("step %d, test accuracy %g, loss %g" % (i, test_accuracy, loss))
#计算精度
print(sess.run(accuracy,feed_dict={x:cifar10.test.images,y_:cifar10.test.labels,keep_prob:1.0}))