import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import freeze_graph

#初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
Saver = tf.train.import_meta_graph('./model/model.meta')
Saver.restore(sess,tf.train.latest_checkpoint('./model/'))
graph_def = tf.get_default_graph().as_graph_def()
output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
		sess,
		graph_def,
		["rgb_input","flow_input","add", 'Softmax'] #需要保存节点的名字
	)
with tf.gfile.GFile('model.pb', "wb") as f:  # 保存模型
	f.write(output_graph_def.SerializeToString())  # 序列化输出



