# Some code for Long Short Term Memory
## Overview
Some implementations of LSTM model, currently including:
* Language modeling on PTB dataset

  An initial Theano implementation of the *sequence-to-sequence* LSTM, based on the torch code https://github.com/wojzaremba/lstm with the same test on PTB dataset. It preserves features including gradient clipping, mini-batch, learning rate adjustment.
  
* Reber Grammar test

  A sample code of mini-batch implementation for the *sequence-to-sequence* LSTM based on the theano code http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php. 
  
  The original code is not suitable running on GPU and with mini-batch, it gives around 4-5 times speedup on CPU version and makes itself GPU friendly. The speedup part is mainly referred to the Theano lstm tutorial code for sentiment analysis (sequence-to-label) http://deeplearning.net/tutorial/code/lstm.py

## Dependencies

* Nvidia Kepler or later GPU with CUDA 6.5 or newer
* Theano https://github.com/Theano/Theano

## Running
For the PTB test:

Download PTB data files (ptb.train.txt,ptb.valid.txt,ptb.test.txt) from the repository https://github.com/wojzaremba/lstm/tree/master/data into the same folder as of the scripts, and run

`$ python lstm_PTB.py`

For the Reber Grammar test:

`$ python lstm_RG.py`

## Notes
0. There are still some issues with the PTB test code, 
namely the perplexity of training set decreases while increasing for validation set, 
also the training speed is slower than torch code with almost the same setting.
For productivity purpose, there are already some theano-based implementations for LSTM model, including

 * Small Theano LSTM module https://github.com/JonathanRaiman/theano_lstm
 * Keras https://github.com/fchollet/keras

1. The current code only implements single LSTM layer in contrast to multiple layers in the original torch code
2. The code has been tested on Ubuntu 14.04, CUDA 6.5/7, GTX Titan Black/980 Ti/Tesla K40m
3. You may need to set the number of splits for the validation set and test set with respect to the VRAM size of your GPU.
