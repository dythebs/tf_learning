import input_data
import tensorflow as tf
#读入数据
train_datas,train_labels,test_datas,test_labels = input_data.read_data_sets('Iris_data/')
#实现模型
x = tf.placeholder(dtype=tf.float32,shape=[None,4])
W = tf.Variable(tf.zeros([4,3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder(dtype=tf.float32,shape=[None,3])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
#评估模型
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#初始化
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
#训练模型
for i in range(3000) :
	sess.run(train_step,feed_dict={x:train_datas,y_:train_labels})

#计算精度
print(sess.run(accuracy,feed_dict={x:test_datas,y_:test_labels}))