import random
import numpy as np
import os
import sys

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split


class ECGDataset(Dataset):
  def __init__(self, metadata, preloaded_data, transform=None, target_transform=None):
    self.metadata = metadata
    self.preloaded_data = preloaded_data
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    print(self.metadata)
    entry = self.metadata[idx]
    class_label = entry['class']

    sig = self.preloaded_data[entry['path']]
    sig = extend_ts(sig, 15000).astype('float32').T

    # Add 50 1's after AF event ends, from trigger word detection example
    # For the contest, wants to be within +/- 1 beat
    # Original signal is 15000 samples, at 200 Hz, so 15000/200 = 75 seconds
    # CNN will encode 15000 samples into a smaller state.
    # 50 1's should cover 3 beats, so that's ~3-5 seconds
    # 50 1's / 4 seconds = 12.5 Hz
    # 75 seconds * 12.5 samples/second = 937
    # So we'll make Y 1000 long
    label = np.zeros(1000).astype('float32')
    if class_label == 1:
      label = insert_ones(label, entry['af_ends'][0])
    
    if self.transform:
      sig = self.transform(sig)

    if self.target_transform:
      label = self.target_transform(label)

    return sig, label

def insert_ones(y, end_step_x):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of end_step should be 0 while, the
    50 following labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    
    end_step_y = int(end_step_x * 1000 / 15000.0)


    # Add 1 to the correct index in the background label (y)
    ### START CODE HERE ### (≈ 3 lines)
    for i in range(end_step_y+1, end_step_y+51):
        if i < y.shape[0]:
            y[i] = 1
    ### END CODE HERE ###
    
    return y

def extend_ts(ts, length):
    extended = np.zeros((length, 2))
    siglength = np.min([length, ts.shape[0]])
    extended[:siglength] = ts[:siglength]
    return extended 