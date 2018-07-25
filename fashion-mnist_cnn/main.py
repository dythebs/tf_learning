import input_data
import tensorflow as tf
#读入数据
mnist = mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#权重初始化
def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(shape=shape,value=0.1)
	return tf.Variable(initial)
#进行一次卷积操作
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#进行一次池化操作
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#实现模型
x = tf.placeholder(dtype=tf.float32,shape=[None,784])
x_image = tf.reshape(x,[-1,28,28,1])
#第一层卷积
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#第二层卷积
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#全连接层
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#输出层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
y_ = tf.placeholder(dtype=tf.float32,shape=[None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#评估模型
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#初始化
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
#训练模型
for i in range(2000) :
	batch_xs,batch_ys = mnist.train.next_batch(100)
	if i % 100 == 0 :
		train_accuracy = sess.run(accuracy,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1.0})
		print("step %d, training accuracy %g" % (i,train_accuracy))
	sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
#计算精度
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))