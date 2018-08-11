import input_data
import numpy as np 
import matplotlib.pyplot as plt
import os


LOG_DIR = 'log/'
SPRITE_FILE = 'mnist_sprite.jpg'
META_FILE = 'mnist_meta.tsv'


def create_sprite_image(images):
	img_h = images.shape[1]
	img_w = images.shape[2]

	m = int(np.ceil(np.sqrt(images.shape[0])))

	sprite_image = np.ones((img_h*m, img_w*m))

	for i in range(m):
		for j in range(m):
			cur = i * m + j
			if cur < images.shape[0]:
				sprite_image[i*img_h:(i+1)*img_h,
							j*img_w:(j+1)*img_w] = images[cur]
			else:
				break

	return sprite_image


#读入数据
mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)


#检查路径是否存在
if not os.path.exists(LOG_DIR):
	os.mkdir(LOG_DIR)


#sprite图像
#原图是黑底白字的，转换一下
images = 1 - mnist.test.images.reshape(-1,28,28)
sprite_image = create_sprite_image(images)
plt.imsave(os.path.join(LOG_DIR, SPRITE_FILE), sprite_image, cmap='gray')


#tsv文件
with open(os.path.join(LOG_DIR, META_FILE), 'w') as fp:
	fp.write('Index\tLabel\n')
	for index, label in enumerate(mnist.test.labels):
		fp.write('%d\t%d\n' % (index, label))

