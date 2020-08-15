"""
COMS 4705 Natural Language Processing Fall 2019
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Main File
Authors: Elsbeth Turcan <eturcan@cs.columbia.edu>

<YOUR NAME HERE> Yu Guo
<YOUR UNI HERE> yg2683
"""

# Imports
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils
import models

# Global definitions - data
DATA_FN = 'data/crowdflower_data.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    ########## YOUR CODE HERE ##########
    # TODO: Given a model, data, and loss function, you should do the following:
    # TODO: 1) Loop through the whole train dataset performing batch optimization with the optimizer of your choice,
    # TODO: updating the model parameters with each batch (we suggest you use torch.optim.Adam to start);
    # TODO: 2) Each time you reach the end of the train dataset (one "epoch"), calculate the loss on the whole dev set;
    # TODO and 3) stop training and return the model once the development loss stops improving (called early stopping).
    # TODO: Make sure to print the dev set loss each epoch to stdout.

    # record current minimum dev loss and set a counter to implement early stopping
    min_loss = np.inf
    stop_count = 0
    dev_loss = 0
    for epoch in range(5000):
        # check for early stop, if dev-loss has stopped decreasing for 10 epochs, stop iterating
        if stop_count == 5:
            print("Dev loss has stopped improving for 5 epochs, executing early stop and returning optimal model.")
            break
        # train model by looping through minibatches
        model.train()
        for batch, (data, target) in enumerate(train_generator, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target.long())
            loss.backward()
            optimizer.step()
        # calculate dev loss and print it out
        model.eval()
        for data, target in dev_generator:
            output = model(data)
            loss = loss_fn(output, target.long())
            dev_loss += loss.item()
        # if dev loss decreases, update min_loss, save model and reset counter, otherwise increase counter
        if dev_loss < min_loss:
            min_loss = dev_loss
            stop_count = 0
            torch.save(model, 'model.pkl')
        else:
            stop_count += 1
        print("Epoch %d:\t dev loss: %.5f \t epochs without improvement: %d" % (epoch, dev_loss, stop_count))
        dev_loss = 0
    return torch.load('model.pkl')
    # raise NotImplementedError


def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


def main():
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")
            

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    ########## YOUR CODE HERE ##########
    # TODO: for each of the two models, you should 1) create it,
    # TODO 2) run train_model() to train it, and
    # TODO: 3) run test_model() on the result

    # build dense nn model
    print("-" * 10 + " Training Dense NN model " + "-" * 10)
    dense_hidden = 100
    DENSE_PATH = 'dense.pth'
    model = models.DenseNetwork(embeddings, dense_hidden, NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters())
    dense_model = train_model(model, loss_fn, optimizer, train_generator, dev_generator)
    torch.save(dense_model, DENSE_PATH)
    test_model(dense_model, loss_fn, test_generator)
    print()
    
    # build rnn model
    print("-" * 10 + " Training RNN model " + "-" * 10)
    rnn_hidden = 25
    RNN_PATH = 'recurrent.pth'
    model = models.RecurrentNetwork(embeddings, rnn_hidden, NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters())
    rnn_model = train_model(model, loss_fn, optimizer, train_generator, dev_generator)
    torch.save(rnn_model, RNN_PATH)
    test_model(rnn_model, loss_fn, test_generator)
    print()

    # build bi-directional LSTM model
    print("-" * 10 + " Training bi-LSTM model " + "-" * 10)
    lstm_hidden = 25
    BILSTM_PATH = 'bi-lstm.pth'
    model = models.BiLstmNetwork(embeddings, lstm_hidden, NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters())
    bilstm_model = train_model(model, loss_fn, optimizer, train_generator, dev_generator)
    torch.save(bilstm_model, BILSTM_PATH)
    test_model(bilstm_model, loss_fn, test_generator)
    print()

    # build Convolution NN model
    print("-" * 10 + " Training CNN model " + "-" * 10)
    num_filters = 100
    CNN_PATH = 'cnn.pth'
    model = models.ConvolutionalNetwork(embeddings, num_filters, NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters())
    cnn_model = train_model(model, loss_fn, optimizer, train_generator, dev_generator)
    torch.save(cnn_model, CNN_PATH)
    test_model(cnn_model, loss_fn, test_generator)



if __name__ == '__main__':
    main()
