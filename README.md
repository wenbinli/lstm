# theano-lstm
## Overview
Some implementations of LSTM model, currently including:
* an initial Theano implementation of the *sequence-to-sequence* LSTM, based on the torch code https://github.com/wojzaremba/lstm with the same test on PTB dataset.
* a sample code of mini-batch implementation for the *sequence-to-sequence* LSTM based on the theano code http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php for Reber Grammar 

## Dependencies

* Nvidia Kepler or later GPU with CUDA 6.5 or newer
* Theano https://github.com/Theano/Theano

## Running
For the PTB test:

`$ python lstm_PTB.py`

For the Reber Grammar test:

`$ python lstm_RG.py`

## Notes
0. There are still some issues with the PTB test code, 
namely the perplexity of training set decreases while increasing for validation set, 
also the training speed is slower than torch code with almost the same setting.
For productivity purpose, there are already quite a few theano-based implementations for LSTM model, including

 * Theano lstm tutorial code for sentiment analysis (sequence-to-label)
 * Small Theano LSTM module https://github.com/JonathanRaiman/theano_lstm
 * Keras https://github.com/fchollet/keras

1. The current code only implements single LSTM layer in contrast to multiple layers in the original torch code
2. The code has been tested on Ubuntu 14.04, CUDA 6.5/7, GTX Titan Black/980 Ti/Tesla K40m
3. You may need to set the number of splits for the validation set and test set with respect to the VRAM size of your GPU.
