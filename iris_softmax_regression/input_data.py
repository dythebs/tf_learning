import os
import numpy
import pandas as pd
from six.moves import urllib
from sklearn.preprocessing import OneHotEncoder

SOURCE_URL = 'http://download.tensorflow.org/data/'
def maybe_download(filename, work_directory):
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath
  
def read_data_sets(train_dir):
  TRAIN = 'iris_training.csv'
  TEST = 'iris_test.csv'
  local_file = maybe_download(TRAIN, train_dir)
  CSV_COLUMN_NAMES = ['SepalLength','SepalWidth','PetalLength', 'PetalWidth', 'Species']
  train = pd.read_csv(local_file,names=CSV_COLUMN_NAMES,header=0)
  local_file = maybe_download(TEST, train_dir)
  test = pd.read_csv(local_file,names=CSV_COLUMN_NAMES,header=0)
  train_datas, train_labels = train.iloc[:,:-1].values, train.iloc[:,-1:].values
  test_datas, test_labels = test.iloc[:,:-1], test.iloc[:,-1:]
  train_labels, test_labels = OneHotEncoder().fit_transform(train_labels).toarray(), OneHotEncoder().fit_transform(test_labels).toarray()
  return train_datas,train_labels,test_datas,test_labels


