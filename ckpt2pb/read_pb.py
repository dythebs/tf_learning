import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


#将保存的模型文件解析为GraphDef  
graph_def = tf.GraphDef()  

model_f = gfile.FastGFile("model.pb",'rb')  
graph_def.ParseFromString(model_f.read())  
tf.import_graph_def(graph_def, name="") #必须要加name 玄学

sess = tf.Session()
sess.run(tf.global_variables_initializer())

model_logits = sess.graph.get_tensor_by_name("add:0")		
model_predictions = sess.graph.get_tensor_by_name("Softmax:0")
rgb_input = sess.graph.get_tensor_by_name("rgb_input:0")
flow_input = sess.graph.get_tensor_by_name("flow_input:0")

#读入数据
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}
rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
flow_sample = np.load(_SAMPLE_PATHS['flow'])
#计算精度
out_logits, out_predictions = sess.run([model_logits, model_predictions],feed_dict={rgb_input:rgb_sample, flow_input:flow_sample})

out_logits = out_logits[0]
out_predictions = out_predictions[0]
sorted_indices = np.argsort(out_predictions)[::-1]
print('Norm of logits: %f' % np.linalg.norm(out_logits))

