"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CNN_RNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # self.layer_1a_size = config["layer_1a_size"]
        # self.layer_1b_size = config["layer_1b_size"]
        # self.layer_1p_size = config["layer_1p_size"]

        # self.layer_2a_size = config["layer_2a_size"]
        # self.layer_2b_size = config["layer_2b_size"]
        # self.layer_2p_size = config["layer_2p_size"]

        self.conv1 = nn.Conv1d(2, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.pool1 = nn.MaxPool1d(3, 3)

        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.pool2 = nn.MaxPool1d(3, 3)

        self.conv5 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv6 = nn.Conv1d(128, 128, 3, padding=1, stride=1)
        self.pool3 = nn.MaxPool1d(3, 3)

        self.fc1 = nn.Linear(71040, 1000)
        self.fc2 = nn.Linear(1000, 1000)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.sigmoid(self.fc2(x))
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
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels)/float(np.sum(mask))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
