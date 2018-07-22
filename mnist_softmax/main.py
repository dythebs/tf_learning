import input_data
import tensorflow as tf
#读入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#实现模型
x = tf.placeholder(dtype=tf.float32,shape=[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder(dtype=tf.float32,shape=[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#评估模型
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuary = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#初始化
init = tf.initialize_all_variables()
sess = tf.Session()
#训练模型
sess.run(init)
for i in range(2000) :
	batch_xs,batch_ys = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
#计算精度
print(sess.run(accuary,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
