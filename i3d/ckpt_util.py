import requests
import os
import tarfile
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf 
import numpy as np 

url ='http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz'
ckpt_path = 'ckpt'

def maybedownload(url):
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
	filename = url.split('/')[-1]
	if not os.path.exists(os.path.join(ckpt_path, filename)):
		with open(os.path.join(ckpt_path, filename), 'wb') as fp:
			fp.write(requests.get(url).content)

	with tarfile.open(os.path.join(ckpt_path, filename)) as fp:
		fp.extractall(path=ckpt_path)

def restore(sess):
	#先检查ckpt文件
	maybedownload(url)
	variables = [v for v in tf.model_variables()]

	#通过2d模型的变量名寻找计算图中对应3d模型的变量
	def get_3d_tensor(name):
		if not name.startswith('global_step') and not name.startswith('InceptionV1/Logits'):
			name_3d = 'InceptionV1_3d' + key[11:]
			for v in variables:
				if v.name.startswith(name_3d):
					return v
		else:
			return None
	
	checkpoint_path = os.path.join(ckpt_path, "inception_v1.ckpt")
	# 读取ckpt文件
	reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
	var_to_shape_map = reader.get_variable_to_shape_map()

	#遍历文件中所有变量
	for key in var_to_shape_map:
		tensor_3d = get_3d_tensor(key)
		if tensor_3d == None:
			continue
		if 'weight' in key:
			weight = reader.get_tensor(key)
			dims = weight.shape[0]
			weight_3d = np.repeat(weight[np.newaxis, :], dims, axis=0) / dims
			sess.run(tf.assign(get_3d_tensor(key), weight_3d))
		else:
			value = reader.get_tensor(key)
			sess.run(tf.assign(get_3d_tensor(key), value))