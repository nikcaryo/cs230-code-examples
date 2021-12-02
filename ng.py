import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torchmetrics
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger

import copy
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
parser.add_argument('--model_path', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

logger = TensorBoardLogger('tb_logs', name='my_model')


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
    mean = np.mean(sig, axis=0, keepdims=True)
    std = np.std(sig, axis=0, keepdims=True)
    sig = (sig - mean) / std
    sig = extend_ts(sig, 15000).astype('float32').T

    # Add 50 1's after AF event ends, from trigger word detection example
    # For the contest, wants to be within +/- 1 beat
    # Original signal is 15000 samples, at 200 Hz, so 15000/200 = 75 seconds
    # CNN will encode 15000 samples into a smaller state.
    # 50 1's should cover 3 beats, so that's ~3-5 seconds
    # 50 1's / 4 seconds = 12.5 Hz
    # 75 seconds * 12.5 samples/second = 937
    # So we'll make Y 1000 long
    label = np.zeros(128).astype('float32')
    
    if class_label == 1:
      label = insert_ones(label, entry['af_ends'][0])

    # if np.random.randn() > 0.5: 
    #     sig = np.flip(sig, 1).copy()
    #     label = np.flip(label).copy()
    
    

    if self.transform:
      sig = self.transform(sig)

    if self.target_transform:
      label = self.target_transform(label)

    end = -1 if not entry['af_ends'] else entry['af_ends'][0]
    return sig, label, end

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
    
    end_step_y = int(end_step_x * 128 / 15000.0)
    # for i in range(end_step_y, end_step_y + 5):
    for i in range(0, end_step_y):
        if i < y.shape[0]:
            y[i] = 1
    
    return y

def extend_ts(ts, length):
    extended = np.zeros((length, 2))
    siglength = np.min([length, ts.shape[0]])
    extended[:siglength] = ts[:siglength]
    return extended 


# https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            print('small len')
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        # print(x_reshape.size())
        y = self.module(x_reshape)
        y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)

        # IF need timesteps first,   We have to reshape Y
        if not  self.batch_first:
            y = y.transpose(0,1).contiguous()  # transpose to (timesteps, samples, output_size)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class CNN_RNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.metrics = {
            'val_loss': [],
            'val_loss_adj': [],
            'train_loss': []
        }
        self.lr = config['lr']
        self.f1 = torchmetrics.F1(num_classes=2, mdmc_average='global')

        self.conv1 = nn.Conv1d(2, 8, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm1d(8)

        self.conv2 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm1d(16)

        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.batch3 = nn.BatchNorm1d(32)

        self.conv4 = nn.Conv1d(32, 32, 3, padding=1, stride=1)

        self.conv5 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.batch5 = nn.BatchNorm1d(64)

        self.conv6 = nn.Conv1d(64, 64, 3, padding=1, stride=1)


        self.conv7 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.batch7 = nn.BatchNorm1d(128)
        self.conv8 = nn.Conv1d(128, 128, 3, padding=1, stride=1)

        self.pool = nn.MaxPool1d(3, 3)
        self.drop = nn.Dropout(0.2)

        # self.conv3 = nn.Conv1d(256, 256, 3, padding=1, stride=1)
        # self.batch2 = nn.BatchNorm1d(32)

        # self.conv5 = nn.Conv1d(128, 128, 3, padding=1, stride=1)
        # self.batch5 = nn.BatchNorm1d(128)
        # self.conv6 = nn.Conv1d(128, 128, 3, padding=1, stride=1)
        # self.batch6 = nn.BatchNorm1d(128)
        # self.pool3 = nn.MaxPool1d(3, 3)

        # self.fc1 = nn.Linear(71040, 1000)
        # self.fc2 = nn.Linear(1000, 1000)

        # self.gru = nn.GRU(555, 555, num_layers = 2, batch_first=True)
        self.f1 = nn.Linear(555, 1)
        self.td = TimeDistributed(self.f1, batch_first=True)
        # self.tdd = nn.Conv2d(1, num_of_output_channels, (555, 1))
        


    def forward(self, x):
        x = F.relu(self.batch1(self.conv1(x)))

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.pool(x)
        x = self.conv3(x)
        
        for _ in range(1):
            x = self.batch3(x)
            x = F.relu(x)
            x = self.drop(x)
            x = self.conv4(x)

        x = self.batch3(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.pool(x)
        x = self.conv5(x)

        x = self.batch5(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.pool(x)
        x = self.conv7(x)

        for _ in range(1):
            x = self.batch7(x)
            x = F.relu(x)
            x = self.drop(x)
            x = self.conv8(x)

        x = self.batch7(x)
        x = F.relu(x)
        x = self.drop(x)
        # x, b = self.gru(x)
        x = self.td(x)
        x = torch.sigmoid(x)
        return x[:,:,0]

    def training_step(self, batch, batch_idx):
        x, y, end = batch
        # print('asdasdf')
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.metrics['train_loss'].append(loss)
        self.metrics['val_loss_adj'].append(self.metrics['val_loss'][-1])


        return loss

    def validation_step(self, batch, batch_idx):
        x, y, end = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, on_step=True)
        self.metrics['val_loss'].append(loss)
        # self.log("val_f1", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)
        # self.log("val_prec", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)
        # self.log("val_rec", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)
        # metrics.append(trainer.callback_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, end = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_f1", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)
        # self.log("val_prec", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)
        # self.log("val_rec", self.f1(y_hat > 0.5, y > 0.5), prog_bar=True)

        return loss

    # def training_epoch_end(self, outputs):
    #     sch = self.lr_schedulers()
    #     # If the selected scheduler is a ReduceLROnPlateau scheduler.
    #     if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #         sch.step(self.trainer.callback_metrics["val_loss"]) 
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     self.logger.experiment.add_scalar()
    #     tensorboard_logs = {'train_loss': avg_loss}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     logger.experiments.add_scalar()
    #     tensorboard_logs = {'val_loss': avg_loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # return optimizer
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 1)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler1,
            #     "monitor": "val_loss",
            #     "frequency": 400
            # }
        }


def get_preloaded_data(metadata_json, limit = None):
    metadata = list(filter(lambda x: x['class'] != 2, metadata_json.values()))
    # metadata = list(filter(lambda x: not x['af_ends'] or x['af_ends'][0] > 200, metadata))
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


def plot(x, y, end, y_hat, i, source, model_path):
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(x[0, 0,:])
    if end != -1:
        plt.axvline(x=end, color='red')
    plt.subplot(2, 1, 2)
    plt.plot(y_hat.detach().numpy()[0,:])
    plt.ylabel('probability')
    os.makedirs(model_path[:-5], exist_ok=True)
    plt.savefig(f'{model_path[:-5]}/plot_{source}_{i}.jpg')
    print('saved', i, source, model_path)
    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading the datasets...")

    with open(args.data_meta_json) as f:
        metadata_json = json.load(f)
        metadata, preloaded_data = get_preloaded_data(metadata_json, limit = args.data_limit)

    SEED = 42
    pl.seed_everything(42, workers=True)
    ecg_data = ECGDataset(metadata, preloaded_data, transform=None)
    train, val, test = get_train_test_split(ecg_data, 0.7, 0.2)
    
    trainer = pl.Trainer(max_epochs=50, gpus=1, log_every_n_steps=1)
    afib = 0
    for x, y, end in DataLoader(ecg_data):
        if end != -1:
            afib += 1 
    logging.info(f"{afib} afib examples.")
    logging.info(len(ecg_data))



    if not args.model_path:
        # Create the input data pipeline
        for i, lr in enumerate([1e-3, 1e-4, 1e-5]):
            cnn_rnn = CNN_RNN({'lr':lr})
            trainer = pl.Trainer(max_epochs=100, gpus=1, log_every_n_steps=1)
            trainer.fit(cnn_rnn, DataLoader(train, batch_size=50, num_workers=4), DataLoader(val, num_workers=4))
            # trainer.test(cnn_rnn, DataLoader(test))
            print(cnn_rnn.metrics)
            train_loss = cnn_rnn.metrics['train_loss']
            val_loss = cnn_rnn.metrics['val_loss_adj']
            plt.clf()
            # plt.subplot(2, 1, 1)
            plt.plot(np.convolve([loss.cpu().detach().numpy() for loss in train_loss][1:-1], np.ones(1)/1), label='Train Loss')
            # plt.subplot(2, 1, 2)
            plt.plot(np.convolve([loss.cpu().detach().numpy() for loss in val_loss][1:-1], np.ones(1)/1), label='Val Loss')
            logging.info(f'Big {lr}, 11110000 format')
            logging.info([x.cpu().detach().numpy() for x in cnn_rnn.metrics['train_loss'][-5:]])
            logging.info([x.cpu().detach().numpy() for x in cnn_rnn.metrics['val_loss_adj'][-5:]])
            plt.xlabel('Batch (over 100 epochs)')
            plt.ylabel('Loss (over 100 epochs)')
            plt.title(f'LR = {lr}')
            plt.savefig(f'x_{i}_lr.png')
    else:
        
        model = CNN_RNN.load_from_checkpoint(args.model_path, config={'lr': 1e-4})
        # results = trainer.test(model, DataLoader(val), verbose=True)
        i = 0
        predictions = []
        for x, y, end in DataLoader(val):
            i += 1
            if i > 20: break
            y_hat = model(torch.tensor(x))
            # predictions.append(y_hat)
            plot(x, y, end, y_hat, i, 'val', args.model_path)

        i = 0
        for x, y, end in DataLoader(train):
            i += 1
            if i > 20: break
            y_hat = model(torch.tensor(x))
            # predictions.append(y_hat)
            plot(x, y, end, y_hat, i, 'train', args.model_path)
        



