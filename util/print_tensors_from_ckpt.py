import os
from tensorflow.python import pywrap_tensorflow

ckpt_name = '"inception_v1.ckpt"'
checkpoint_path = os.path.join(ckpt_name)
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
	print("tensor_name: ", key)
	print(type(reader.get_tensor(key)))