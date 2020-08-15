# HW2: Multilabel Classification with Neural Network
In hw2.py, main() function trains and tests all four models - two basic ones and two extensions.
train_model() performs training on minibatches, prints dev loss each epoch, keeps track of current optimal model, implements early stopping once model ceases to improve for 5 epochs and returns last saved model.

All 4 model classes are in models.py, and the trained models are all included.

Basic dense model as "dense.pth", test F1 score is 42.86
Basic RNN model as "recurrent.pth", test F1 score is 42.19
Extension bi-LSTM model as "bi-lstm.pth", test F1 score is 46.32
Extension CNN model as "cnn.pth", test F1 score is 48.25
