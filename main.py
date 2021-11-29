import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torchmetrics

import wfdb

import os
import argparse
import logging
import json

import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data_meta_json', default='data/physionet_meta_relative.json',
                    help="JSON file containing where each individual signal/label lives.")
parser.add_argument('--data_limit', default=None, type=int,
                    help="Maximum number of examples to use (default: all)")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

class ECGDataset(Dataset):
  def __init__(self, metadata, preloaded_data, transform=None, target_transform=None):
    self.metadata = metadata
    self.preloaded_data = preloaded_data
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
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
    for i in range(end_step_y+1, end_step_y+51):
        if i < y.shape[0]:
            y[i] = 1
    
    return y

def extend_ts(ts, length):
    extended = np.zeros((length, 2))
    siglength = np.min([length, ts.shape[0]])
    extended[:siglength] = ts[:siglength]
    return extended 


class CNN_RNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # self.layer_1a_size = config["layer_1a_size"]
        # self.layer_1b_size = config["layer_1b_size"]
        # self.layer_1p_size = config["layer_1p_size"]

        # self.layer_2a_size = config["layer_2a_size"]
        # self.layer_2b_size = config["layer_2b_size"]
        # self.layer_2p_size = config["layer_2p_size"]
        self.f1 = torchmetrics.F1(num_classes=2, mdmc_average='global')
        self.conv1 = nn.Conv1d(2, 8, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(3, 3)

        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.batch3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.batch4 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(3, 3)

        self.conv5 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.batch5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 128, 3, padding=1, stride=1)
        self.batch6 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(3, 3)

        self.fc1 = nn.Linear(71040, 1000)
        self.fc2 = nn.Linear(1000, 1000)

        self.lstm = nn.LSTM(1666, 1000, num_layers = 2)

    def forward(self, x):
        x = F.relu(self.batch1(self.conv1(x)))
        x = F.relu(self.batch2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.batch3(self.conv3(x)))
        x = F.relu(self.batch4(self.conv4(x)))
        x = self.pool2(x)
        x = F.relu(self.batch5(self.conv5(x)))
        x = F.relu(self.batch6(self.conv6(x)))
        x = self.pool3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        # x, (hn, cn) = self.lstm(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)
        # self.log("val_prec", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)
        # self.log("val_rec", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_f1", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)
        # self.log("val_prec", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)
        # self.log("val_rec", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def get_preloaded_data(metadata_json, limit = None):
    print(metadata_json.values())
    metadata = list(filter(lambda x: x['class'] != 2, metadata_json.values()))
    limit = len(metadata) if limit is None else limit
    preloaded_data = {}
    sampto = 50000
    for entry in metadata[:limit]:
        sig, _ = wfdb.rdsamp(entry['path'], sampto=min(sampto, entry['sig_len']))
        preloaded_data[entry['path']] = sig
    return metadata[:limit], preloaded_data

def get_train_test_split(ecg_data, percent_train, percent_val):
    m_train = int(percent_train * len(ecg_data))
    m_val = int(percent_val * len(ecg_data))
    m_test = len(ecg_data) - m_train - m_val
    return random_split(ecg_data, [m_train, m_val, m_test])


if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    with open(args.data_meta_json) as f:
        metadata_json = json.load(f)
        metadata, preloaded_data = get_preloaded_data(metadata_json, limit = args.data_limit)

    SEED = 42
    pl.seed_everything(42, workers=True)
    ecg_data = ECGDataset(metadata, preloaded_data, transform=None)
    train, val, test = get_train_test_split(ecg_data, 0.7, 0.2)
    cnn_rnn = CNN_RNN({})
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(cnn_rnn, DataLoader(train, batch_size=10), DataLoader(val))
    trainer.test(cnn_rnn, DataLoader(test))
