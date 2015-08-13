#!/usr/bin/python
import numpy as np
import theano
import theano.tensor as T
import reberGrammar

dtype = theano.config.floatX

# SET the random number generator's seeds for consistency
SEED = 123
np.random.seed(SEED)

# refer to the tutorial 
# http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php
# http://deeplearning.net/tutorial/code/lstm.py

# activation function for others
tanh = T.tanh
# activation function for gates
sigma = lambda x: 1 / (1 + T.exp(-x))

# lstm unit - extended version include forget gate and peephole weights
def lstm_step(x_t,m_t,h_tm1,c_tm1, # changes: add m_t for mask variable at time step t
              W_x,W_h,W_c,W_co,W_hy,
              b_i,b_f,b_c,b_o,b_y):
    
    h_dim = h_tm1.shape[-1] # hidden unit dimension
        
    def _slice(_x,n,dim):
        return _x[:,n * dim:(n + 1) * dim]
            
    # improve efficiency
    preact_x = T.dot(x_t,W_x)
    preact_h = T.dot(h_tm1,W_h)
    preact_c = T.dot(c_tm1,W_c)
                                  
    # input gate
    i_t = T.nnet.sigmoid(_slice(preact_x,0,h_dim) + _slice(preact_h,0,h_dim)  + _slice(preact_c,0,h_dim) + b_i)
    # forget gate
    f_t = T.nnet.sigmoid(_slice(preact_x,1,h_dim) + _slice(preact_h,1,h_dim) + _slice(preact_c,1,h_dim) + b_f)
    # cell
    c_t = f_t * c_tm1 + i_t * tanh(_slice(preact_x,3,h_dim) + _slice(preact_h,3,h_dim) + b_c)
    c_t = m_t[:,None] * c_t + (1. - m_t)[:,None] * c_tm1 # add mask
    
    # output gate
    o_t = T.nnet.sigmoid(_slice(preact_x,2,h_dim) + _slice(preact_h,2,h_dim ) + T.dot(c_t,W_co) + b_o)    

    # cell output
    h_t = o_t * tanh(c_t)
    h_t = m_t[:,None] * h_t + (1. - m_t)[:,None] * h_tm1 # add mask
    # output
    y_t = T.nnet.sigmoid(theano.dot(h_t,W_hy) + b_y)

    return [h_t,c_t,y_t]

# random initialization of weights
def init_weights(size_x,size_y):
    values = np.ndarray([size_x,size_y],dtype=dtype)
    for dx in xrange(size_x):
        vals = np.random.uniform(low=-1.,high=1.,size=(size_y,))
        values[dx,:] = vals
    _,svs,_ = np.linalg.svd(values)
    # svs[0] is the largest singular value
    values = values / svs[0]
    return values

# get minibatches' index and shuffle the dataset at each iteration, taken from the lstm.py
def get_minibatches_idx(n,minibatch_size, shuffle=False):
    idx_list = np.arange(n,dtype="int32")
    
    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range( n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):# make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    return zip(range(len(minibatches)),minibatches)

# claim numpy array object
def numpy_floatX(data):
    return np.asarray(data, dtype=dtype)

#------------------ test case -----------------------
# instantiate a lstm network for reber grammar
n_in = 7
n_hidden = n_i = n_c = n_o = n_f = 10
n_y = 7

# initialize weights
W_x = theano.shared(init_weights(n_in,n_hidden*4))
W_h = theano.shared(init_weights(n_hidden,n_hidden*5))
W_c = theano.shared(init_weights(n_hidden,n_hidden*2))
W_co = theano.shared(init_weights(n_hidden,n_hidden))
W_hy = theano.shared(init_weights(n_hidden, n_y))

b_i = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size=n_i)))
b_f = theano.shared(np.cast[dtype](np.random.uniform(0,1.,size=n_f)))
b_c = theano.shared(np.zeros(n_c,dtype=dtype))
b_o = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size=n_o)))
b_y = theano.shared(np.zeros(n_y,dtype=dtype))

params = [W_x,W_h,W_c,W_co,W_hy,b_i,b_f,b_c,b_o,b_y]

# input
v = T.tensor3(dtype=dtype)
n_samples = v.shape[1]
# mask
m = T.matrix(dtype=dtype)
target = T.tensor3(dtype=dtype)

# sequential model
[h_vals,_,y_vals],_ = theano.scan(fn = lstm_step,
                                  sequences = [v,m],
                                  outputs_info = [T.alloc(numpy_floatX(0.),n_samples,n_hidden),
                                                  T.alloc(numpy_floatX(0,),n_samples,n_hidden),None],
                                  non_sequences = [W_x,W_h,W_c,W_co,W_hy,b_i,b_f,b_c,b_o,b_y])

# cost
cost = -T.mean(target * T.log(y_vals) + (1. - target) * T.log(1. - y_vals))

# learning rate
lr = np.cast[dtype](.1)
learning_rate = theano.shared(lr)

gparams = []
for param in params:
    gparam = T.grad(cost,param)
    gparams.append(gparam)

updates = []
for param,gparam in zip(params,gparams):
    updates.append((param,param - gparam * learning_rate))

#---------------- change data format and padding
# generate data
train_data = reberGrammar.get_n_embedded_examples(1000)

num_samples = len(train_data)

lengths = [] #counter for sequence length
for j in range(len(train_data)):
    i,o = train_data[j]
    lengths.append(len(i))

maxlen = max(lengths)
# zero padding by the maximum length of seqs
train_input = np.zeros((maxlen,num_samples,n_in),dtype=np.float32)
train_mask = np.zeros((maxlen,num_samples),dtype=np.float32)
train_tgt = np.zeros((maxlen,num_samples,n_in),dtype=np.float32)

for j in range(num_samples):
    i,o = train_data[j]
    train_input[:lengths[j],j] = np.vstack(i)
    train_tgt[:lengths[j],j] = np.vstack(o)
    train_mask[:lengths[j],j] = 1

#----------------------------------------------------

learn_rnn_fn = theano.function(inputs = [v,m,target],
                               outputs = cost,
                               updates = updates)

#-----------------Apply minibatch 
nb_epochs = 250
batch_size = 50 # mini-batch size
train_err = np.ndarray(nb_epochs)
def train_rnn(train_data):
    for epo in range(nb_epochs):
        print "training epoch ",str(epo),"..."
        error = 0.
        kf = get_minibatches_idx(num_samples,batch_size,shuffle=True)
        for _,train_idx in kf:
            x = train_input[:,train_idx,:]
            y = train_tgt[:,train_idx,:]
            m = train_mask[:,train_idx]

            train_cost = learn_rnn_fn(x,m,y) # modified function
            error += train_cost    

        train_err[epo] = error

train_rnn(train_data)
#-----------------------------------------------------

# plot results
import matplotlib.pyplot as plt
plt.plot(np.arange(nb_epochs),train_err,'b-')
plt.xlabel('epochs')
plt.ylabel('error')
plt.ylim(0.50)

