import os
import numpy
import pandas as pd
from six.moves import urllib
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tarfile
import pickle
import numpy as np

SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'
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

  TARGET = 'cifar-10-python.tar.gz'
  local_file = maybe_download(TARGET, train_dir)
  with tarfile.open(local_file) as tar:
    names = tar.getnames()
    for name in names:
      print('Extracting', name)
      tar.extract(name,path=train_dir)
  META = 'batches.meta'
  TRAIN = 'data_batch_'
  TEST = 'test_batch'
  train_datas = []
  train_labels = []
  test_datas = []
  test_labels = []
  for i in range(5):
    dict = unpickle(os.path.join(train_dir,'cifar-10-batches-py',TRAIN+str(i+1)))
    train_datas.append(dict[b'data'])
    train_labels.append(dict[b'labels'])
  dict = unpickle(os.path.join(train_dir,'cifar-10-batches-py',TEST))
  test_datas.append(dict[b'data'])
  test_labels.append(dict[b'labels'])

  data_sets.train = DataSet(np.array(train_datas).reshape(50000,3072), np.array(train_labels))
  data_sets.test = DataSet(np.array(test_datas).reshape(10000,3072), np.array(test_labels))

  '''
  CSV_COLUMN_NAMES = ['SepalLength','SepalWidth','PetalLength', 'PetalWidth', 'Species']
  train = pd.read_csv(local_file,names=CSV_COLUMN_NAMES,header=0)
  local_file = maybe_download(TEST, train_dir)
  test = pd.read_csv(local_file,names=CSV_COLUMN_NAMES,header=0)
  train_datas, train_labels = train.iloc[:,:-1].values, train.iloc[:,-1:].values
  test_datas, test_labels = test.iloc[:,:-1], test.iloc[:,-1:]
  train_labels, test_labels = OneHotEncoder().fit_transform(train_labels).toarray(), OneHotEncoder().fit_transform(test_labels).toarray()
  '''

  return data_sets