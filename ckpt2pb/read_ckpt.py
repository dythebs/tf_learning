import numpy as np
import tensorflow as tf
#读入数据
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}
rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
flow_sample = np.load(_SAMPLE_PATHS['flow'])


#初始化
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
Saver = tf.train.import_meta_graph('./model/model.meta')
Saver.restore(sess,tf.train.latest_checkpoint('./model/'))

#计算精度
out_logits, out_predictions = sess.run(['add:0', 'Softmax:0'],feed_dict={'rgb_input:0':rgb_sample, 'flow_input:0':flow_sample})

out_logits = out_logits[0]
out_predictions = out_predictions[0]
sorted_indices = np.argsort(out_predictions)[::-1]
print('Norm of logits: %f' % np.linalg.norm(out_logits))
