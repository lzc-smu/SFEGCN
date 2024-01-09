from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .pat import Dataset

dataset_factory = {
  'pat': Dataset,
}


def get_dataset(dataset):
  return dataset_factory[dataset]
