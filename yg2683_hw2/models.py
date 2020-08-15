"""
COMS 4705 Natural Language Processing Fall 2019
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Models File
Authors: Elsbeth Turcan <eturcan@cs.columbia.edu>

<YOUR NAME HERE> Yu Guo
<YOUR UNI HERE> yg2683
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class DenseNetwork(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_dim):
        super(DenseNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.hidden_dim = hidden_dim
        self.embed_size = embeddings.size(1)
        self.embedding = nn.Embedding.from_pretrained(embeddings.float())
        self.embedding.weight.requires_grad = False
        self.dense1 = nn.Linear(self.embed_size, self.hidden_dim)
        self.dense2 = nn.Linear(self.hidden_dim, output_dim)
        # raise NotImplementedError

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Take the sum of all word embeddings in a sentence; and
        # TODO: 3) Feed the result into 2-layer feedforward network which produces a 4-vector of values,
        # TODO: one for each class
        embedded = self.embedding(x)
        summed = torch.sum(embedded, 1)
        return self.dense2(self.dense1(summed))
        # raise NotImplementedError


class RecurrentNetwork(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_dim):
        super(RecurrentNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.embed_size = embeddings.size(1)
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(embeddings.float())
        self.embedding.weight.requires_grad = False
        self.rnn = nn.RNN(self.embed_size, self.hidden_dim, num_layers=2, batch_first=True)
        self.dense = nn.Linear(self.hidden_dim, output_dim)
        # raise NotImplementedError

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class
        seq_len = []
        for i in x:
            seq_len.append(i.nonzero().size(0))
        embedded = self.embedding(x)
        packed_input = rnn.pack_padded_sequence(embedded, seq_len, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed_input)
        output = self.dense(hidden[-1])
        return output
        # raise NotImplementedError


# TODO: If you do any extensions that require you to change your models, make a copy and change it here instead.
# TODO: PyTorch unfortunately requires us to have your original class definitions in order to load your saved
# TODO: dense and recurrent models in order to grade their performance.
# extension-grading!!!
# Extension Experiment 1. Bi-directional LSTM implementation
class BiLstmNetwork(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_dim):
        super(BiLstmNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        self.embed_size = embeddings.size(1)
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(embeddings.float())
        self.embedding.weight.requires_grad = False
        self.bi_LSTM = nn.LSTM(self.embed_size, self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(self.hidden_dim, output_dim)
        # raise NotImplementedError

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        seq_len = []
        for i in x:
            seq_len.append(i.nonzero().size(0))
        embedded = self.embedding(x)
        packed_input = rnn.pack_padded_sequence(embedded, seq_len, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.bi_LSTM(packed_input)
        output = self.dense(hidden[-1])
        return output
        # raise NotImplementedError


# extension-grading!!!
# Extension Experiment 2. CNN implementation
class ConvolutionalNetwork(nn.Module):
    def __init__(self, embeddings, filter_num, output_dim):
        super(ConvolutionalNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        self.embed_size = embeddings.size(1)
        self.embedding = nn.Embedding.from_pretrained(embeddings.float())
        self.embedding.weight.requires_grad = False
        self.conv = nn.Conv1d(self.embed_size, filter_num, 3)
        self.dense = nn.Linear(filter_num, output_dim)
        # raise NotImplementedError

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        embedded = self.embedding(x).permute(0, 2, 1)
        conved = self.conv(embedded).permute(0, 2, 1)
        pooled = conved.max(1)[0]
        output = self.dense(pooled)
        return output
        # raise NotImplementedError
