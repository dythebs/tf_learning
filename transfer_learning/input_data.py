import os
import numpy
import pandas as pd
from six.moves import urllib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tarfile
import pickle
import cv2
import numpy as np

SOURCE_URL = 'http://download.tensorflow.org/examples_images/flower_photos.tgz'

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def maybe_download(filename, work_directory):
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath
  
class DataSet(object):
  def __init__(self, images, labels):

    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images
  @images.setter
  def images(self, value):
    self._images = value
  @property
  def labels(self):
    return self._labels
  @labels.setter
  def labels(self, value):
    self._labels = value
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir):
  class DataSets(object):
    pass
  data_sets = DataSets()

  TARGET = 'flower_photos.tgz'
  local_file = maybe_download(TARGET, train_dir)
  with tarfile.open(local_file) as tar:
    names = tar.getnames()
    for name in names:
      print('Extracting', name)
      tar.extract(name,path=train_dir)

  datas = []
  labels = []

  for dir in os.listdir(os.path.join(train_dir, 'flower_photos')):
    if not os.path.isdir(os.path.join(train_dir, 'flower_photos', dir)):
    	continue
    for image in os.listdir(os.path.join(train_dir, 'flower_photos', dir)):
      abs_path = os.path.join(train_dir, 'flower_photos', dir, image)
      img = cv2.cvtColor(cv2.imread(abs_path), cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, (224,224))
      datas.append(img)
      labels.append(dir)

  datas = np.array(datas) / 255.
  labels = np.array(labels)
  labels = LabelEncoder().fit_transform(labels).reshape(-1,1)
  labels = OneHotEncoder().fit_transform(labels).toarray()
  X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2, random_state=1337)


  data_sets.train = DataSet(X_train, y_train)
  data_sets.test = DataSet(X_test, y_test)

  return data_sets