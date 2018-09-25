import inception_v1
import input_data
import tensorflow.contrib.slim as slim
import tensorflow as tf 


x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
keep_prob = tf.placeholder(dtype=tf.float32)
logits = inception_v1.inception_v1(x, keep_prob, 5)
logits = tf.reshape(logits, [-1, 5])

exclusions = ['InceptionV1/Logits']
inception_except_logits = slim.get_variables_to_restore(exclude=exclusions)
CKPT_FILE = 'inception_v1.ckpt'
init_fn = slim.assign_from_checkpoint_fn(
	CKPT_FILE,
	inception_except_logits, ignore_missing_vars=True)


y = tf.nn.softmax(logits)
y_ = tf.placeholder(dtype=tf.float32,shape=[None, 5])
output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionV1/Logits')
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=output_vars)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

flower_photos = input_data.read_data_sets('flower_photos/')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	init_fn(sess)

	for i in range(3000):
		batch_xs, batch_ys = flower_photos.train.next_batch(128)
		sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.8})
		if i % 100 == 0:
			test_batch_xs, test_batch_ys = flower_photos.test.next_batch(200)
			test_accuracy, loss = sess.run([accuracy, cross_entropy] ,feed_dict={x:test_batch_xs,y_:test_batch_ys,keep_prob:1})
			print("step %d, test accuracy %g, loss %g" % (i, test_accuracy, loss))

	#计算精度
	print(sess.run(accuracy,feed_dict={x:flower_photos.test.images,y_:flower_photos.test.labels,keep_prob:1}))