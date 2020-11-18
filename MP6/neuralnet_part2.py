# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import ceil


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output




        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size = 3, padding = 1),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(6, 12, kernel_size = 3, padding = 1),
            nn.MaxPool2d(2, stride = 2),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, out_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lrate, weight_decay=0.0001)


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = x.reshape(-1, 3, 32, 32)
        return self.model(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach().cpu().numpy()



def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N

    model's performance could be sensitive to the choice of learning_rate. We recommend trying different values in case
    your first choice does not seem to work well.
    """
    mu = torch.mean(train_set)
    sigma = torch.std(train_set)
    train_set = (train_set - mu) / sigma

    mu = torch.mean(dev_set)
    sigma = torch.std(dev_set)
    dev_set = (dev_set - mu) / sigma

    net = NeuralNet(0.01, nn.CrossEntropyLoss(), 3072, 2)

    losses = []
    num_batch = ceil(1.0 * len(train_set) / batch_size)
    for epoch in range(n_iter // batch_size):
        print("epoch:", epoch)
        tmp_loss = []
        for b in range(num_batch):
            x_batch, y_batch = train_set[batch_size * b : min(batch_size * (b + 1), len(train_set)),], train_labels[batch_size * b : min(batch_size * (b + 1), len(train_labels)),]
            loss = net.step(x_batch, y_batch)
            tmp_loss.append(loss * y_batch.shape[0])
        losses.append(np.mean(tmp_loss))
        # print(np.mean(tmp_loss))
    yhats = np.argmax(net.forward(dev_set).detach().cpu().numpy(), axis = 1)
    # print(yhats)
    return losses, yhats, net
