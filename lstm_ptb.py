#!/usr/bin/python

"""
lstm
(1) test full structure (peephole weight?) with batch computation using CPU/GPU
(2) clean the code and try to provide general interface
    a) separate the data interface, data format()
    b) modularized parameter interface

the original code refers to the following tutorial 
  http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php (full mode)
  http://deeplearning.net/tutorial/code/lstm.py (without peephole weights and simplified connections)
copyright@ Wenbin Li (wenbinli@mpi-inf.mpg.de)
"""
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import reberGrammar
import timeit

dtype = theano.config.floatX

# SET the random number generator's seeds for consistency
SEED = 123
np.random.seed(SEED)

# lstm class
class lstm(object):
    """ Long-Short Term Memory RNN class
    
    Supported output types:
    Sequence classification (single label)
    Sequence label/Classification (label per element)
    
    NOTE: currently no mask, compare to the reference code for sentiment analysis
    """

    
    def __init__(self,x,y,n_in,n_hidden,n_out,n_steps,learning_rate,val,max_grad_norm):
        """ 
        constructor
    
        Parameters
        ----------
        x: a symbolic tensor of shape (n_steps,n_samples,n_in) (theano.tensor.d/fmatrix)
        y: a symbolic tensor of shape (n_steps,n_samples,n_in) (theano.tensor.d/fmatrix)
        Change y: a symboic matrix of shape (n_samples,n_steps)
        NOTE: constructutor construct the model for training data (with input & output)

        n_in: dimensionality of input (int)
        n_hidden: dimensionality of hidden unit (int)
        n_out: dimensionality of output (int)
        n_steps: length of the recurrent lstm network
        learning_rate: a symbolic scalar variable for learning rate
        val: initial value for parameters
        max_grad_norm: value cap for the gradient/gradient clipping
        """
        # self.input = input
        
        # initialize weight
        self.W_x = theano.shared(self.init_weights(n_in,n_hidden*4,'uniform',val))
        self.W_h = theano.shared(self.init_weights(n_hidden,n_hidden*5,'uniform',val))
        #self.W_c = theano.shared(self.init_weights(n_hidden,n_hidden*2,'uniform',val))
        #self.W_co = theano.shared(self.init_weights(n_hidden,n_hidden,'uniform',val))
        self.W_hy = theano.shared(self.init_weights(n_hidden, n_out,'uniform',val))

        # initialize bias ? somehow this is not dealt in tutorial code
        self.b_i = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size=n_hidden)))
        self.b_f = theano.shared(np.cast[dtype](np.random.uniform(0,1.,size=n_hidden)))
        self.b_c = theano.shared(np.zeros(n_hidden,dtype=dtype))
        self.b_o = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size=n_hidden)))
        self.b_y = theano.shared(np.zeros(n_out,dtype=dtype))

        # shared variable for the network's parameter        
        #self.params = [self.W_x,self.W_h,self.W_c,self.W_co,self.W_hy,
        self.params = [self.W_x,self.W_h,self.W_hy,
                       self.b_i,self.b_f,self.b_c,self.b_o,self.b_y]

        """
        Define lstm network
        """        
        n_samples = x.shape[1]
        
        def _slice(_x,n,dim):
            return _x[:,n * dim:(n + 1) * dim]
        
        def _step(x_t,h_tm1,c_tm1, 
                  W_x,W_h,W_hy,
                  #W_x,W_h,W_hy,W_c,W_co,
                  b_i,b_f,b_c,b_o,b_y):
                  
            h_dim = h_tm1.shape[-1] # hidden unit dimension
            
            preact_x = T.dot(x_t,W_x)
            preact_h = T.dot(h_tm1,W_h)
                                  
            # input gate
            i_t = T.nnet.sigmoid(_slice(preact_x,0,h_dim) + _slice(preact_h,0,h_dim) + b_i)
            # forget gate
            f_t = T.nnet.sigmoid(_slice(preact_x,1,h_dim) + _slice(preact_h,1,h_dim) + b_f)
            # output gate
            o_t = T.nnet.sigmoid(_slice(preact_x,2,h_dim) + _slice(preact_h,2,h_dim) + b_o)
    
            # cell unit
            c_t = f_t * c_tm1 + i_t * T.tanh(_slice(preact_x,3,h_dim) + _slice(preact_h,3,h_dim) + b_c)
            # c_t = m_t[:,None] * c_t + (1. - m_t)[:,None] * c_tm1 # add mask

            # cell output
            h_t = o_t * T.tanh(c_t)
            
            #if mask_flag == True:
            #    h_t = m_t[:,None] * h_t + (1. - m_t)[:,None] * h_tm1 # add mask
            
            # output, probabilistic output, need to change structure of the network
            # original network for the Reber Grammar uses a sigmoid layer but for PTB-torch test, it uses a softmax
            # y_t = T.nnet.sigmoid(theano.dot(h_t,W_hy) + b_y)
            y_t = theano.dot(h_t,W_hy) + b_y # pre-activation

            return [h_t,c_t,y_t]
            
        # unrolled model
        # h_vals: symbolic tensor for hidden units
        # y_vals: symbolic tensor for output units' pre-activation
        [h_vals,_,y_vals],_ = theano.scan(fn = _step, sequences = [x],
                                  outputs_info = [T.alloc(self.numpy_floatX(0.),n_samples,n_hidden),
                                                  T.alloc(self.numpy_floatX(0,),n_samples,n_hidden),None],
                                  #non_sequences = [self.W_x,self.W_h,self.W_c,self.W_co,self.W_hy,
                                  non_sequences = [self.W_x,self.W_h,self.W_hy,
                                                   self.b_i,self.b_f,self.b_c,self.b_o,self.b_y],
                                  n_steps=n_steps)

        """
        Define output and cost function
        """
        # logsoftmax output unit as in the torch code
        # as note in the rnn-theano code, T.nnet.softmax will not operate on T.tensor3 types, only matrices
        # We take our n_steps x n_seq x n_classes output from the net
        # and reshape it into a (n_steps * n_seq) x n_classes matrix
        # apply softmax, then reshape back
        
        y_p_m = T.reshape(y_vals, (y_vals.shape[0] * y_vals.shape[1], -1))
        y_p_s = T.nnet.softmax(y_p_m)
        # y_p_s = T.reshape(y_p_s,y_vals.shape) # reshape back to n_steps x n_seq x n_classes tensor
        
        # for the PTB language model, the cost is the perplexity
        # refer to G. Taylor's mini-batch rnn theano code
        # cost = -T.mean(T.log(y_p_s * y)) # already averaged over words and mini-batches
        y_f = y.flatten(ndim=1) # y is n_seq x n_steps
        cost = -T.mean(T.log(y_p_s)[T.arange(y_p_s.shape[0]),y_f])
        # cost = -T.mean(y * T.log(y_vals) + (1. - y) * T.log(1. - y_vals)) # old cost function                         

        # Compute gradient for parameters
        gparams = []
        #i = 0
        print "computing gradient ..."
        for param in self.params:
            #print "compute derivative wrt to",str(i),'parameter'
            #i = i + 1
            gparam = T.grad(cost,param)
            gparams.append(gparam)
            
        # define l2 norm of graident of parameters for monitoring
        norm_gparams = 0
        for gparam in gparams:
            norm_gparams += T.sqrt(T.sum(gparam ** 2)) # symbolic variable

        """
        (Stochastic) gradient descent
        DOING: add shrink_factor as in the torch code to boost the performance
        """
        # function computes gradients for given data (mostly as mini-batch)
        # without updating the weights and return the corresponded cost
        
        gshared = [theano.shared(p.get_value() * 0.) for p in self.params]
        
        ####################################################################
        # Shrink factor: IF gparams.norm > params.max_grad_norm THEN
        #                    shrink_factor = params.max_grad_norm/gparams.norm
        #                    gparams = gparams * shrink_factor
        ####################################################################
        
        # use condition syntax to compare symbolic variable to a value
        shrink_factor = ifelse(T.gt(norm_gparams,max_grad_norm),max_grad_norm/norm_gparams,1.)
        """
        if norm_gparams > max_grad_norm: 
            shrink_factor = max_grad_norm/norm_gparams
        else:
            shrink_factor = 1.
        """            
        gup = [(gs,g*shrink_factor) for gs,g in zip(gshared,gparams)] # gradient clipping
        
        print "compiling f_show_grad ..."    
        self.f_show_grad = theano.function([x,y],norm_gparams)
               
        # gup = [(gs,g) for gs,g in zip(gshared,gparams)] # gradient updates
        
        #for param,gparam in zip(self.params,gparams):
        #    gup.append((theano.shared(param.get_value()*0.),gparam))
        
        print "compiling f_grad_shared ..."
        self.f_grad_shared = theano.function([x,y],cost,updates=gup)
        
        
        # function updates the weights from the previously computed gradient
        pup = [(p,p - learning_rate * g) for p,g in zip(self.params,gshared)]
        
        #pup = [] # parameter upadtes
        #for param,gparam in zip(self.params,gparams):
        #    pup.append((param,param - gparam * learning_rate))
        print "compiling f_update ..."
        self.f_update = theano.function([learning_rate],[],updates=pup)
        
        
        print "compiling f_cost"
        # function to compute lost
        self.f_cost = theano.function([x,y],cost) # only compute cost without update the shared parameters
        
        
        print "compiling f_pred_prob"
        # function to compute feedforward pass
        self.f_pred_prob = theano.function([x],y_vals)
                    
    # utility function to create numpy floatX object
    def numpy_floatX(self,data):
        return np.asarray(data, dtype=dtype)
    
    # utility function to initialize weights
    # add initialize weight method, the original torch code use uniform
    def init_weights(self,size_x,size_y,method='uniform',init_v=0.1):
        if method == 'uniform':
            values = np.random.uniform(-init_v,init_v,(size_x,size_y))
            values = self.numpy_floatX(values)
        else:
            values = np.ndarray([size_x,size_y],dtype=dtype)
            for dx in xrange(size_x):
                vals = np.random.uniform(low=-1.,high=1.,size=(size_y,))
                values[dx,:] = vals
            _,svs,_ = np.linalg.svd(values)
            # svs[0] is the largest singular value
            values = values / svs[0]
        
        return values

#------------------ test case -----------------------

# utility function for loading the corpus data
# try to match the setting in the torch counterpart
# rewrite from the original torch function
def loader(file_path):
    # read the file as a long and continuous string
    with open(file_path,"r") as f:
        data = f.read().replace('\n','<eof>')
        
    # split the string into a word list by the space
    data_ = data.split()

    # convert the word list into a vocab-index list
    vocab_map = {}
    vocab_idx = 0
    x = np.zeros(len(data_),)
    
    for i in range(len(data_)):
        if not(data_[i] in vocab_map):
            vocab_map[data_[i]] = vocab_idx
            vocab_idx = vocab_idx + 1 # start from 0 instead of 1
        x[i] = vocab_map[data_[i]]
        
    return x
    
# utility function to reshape the steam of words into fixed length seqs
# rewrite from the original torch function
# batch_size: mini-batch size
# return 
#   x_, input sequence, a matrix of (seq_len,num_seqs) 
#             where each column is a newly formed seq
#   y_, output sequence, a matrix of (seq_len,num_seqs) 
#             note the final word is the first word from next seq

def make_seq(x,seq_len):
    w_count = x.shape[0]
    num_seqs = np.ceil(w_count/seq_len).astype(int)
    x_ = np.zeros((seq_len,num_seqs))
    y_ = np.zeros((seq_len,num_seqs))
    
    for i in range(num_seqs):
        start = i * seq_len
        finish = (i+1) * seq_len
        x_[:,i] = x[start:finish]
        if finish == num_seqs * seq_len: # if reach the end of the final seq
            y_[:,i] = np.append(x[start+1:finish],x[-1])
        else:    
            y_[:,i] = x[start+1:finish + 1]
        
    return x_.astype('int32'),y_.astype('int32')
    
# convert mini-batch data into one-hot vectors for each entry
# for each mini-batch (seq_len,num_samples,vocab_size)
# Note this has to be adjusted wrt different VRAM size in GPU instance
#   mat_idx: a matrix of (seq_len,num_seqs) for a mini-batch, each column is a seq
#   return a tensor of (seq_len,num_seqs,vocab_size) of one-hot representation

# UPDATE: try to_one_hot by defining a global function
idx = T.vector(dtype='int32')
num_class = T.scalar(dtype='int32')
conv = theano.function([idx,num_class],T.extra_ops.to_one_hot(idx,num_class,dtype=dtype))

def convert_data(mat_idx,vocab_size):
    seq_len = mat_idx.shape[0]
    num_seqs = mat_idx.shape[1]
    tensor_idx = np.zeros((seq_len,num_seqs,vocab_size)).astype(dtype)
    
    for i in range(num_seqs):
        tensor_idx[:,i,:] = conv(mat_idx[:,i],vocab_size) 
        # not sure if this will speedup or overhead to slow down
    """
    for i in range(num_seqs):
        for j in range(seq_len):
            tensor_idx[j,i,mat_idx[j,i]] = 1
    """
        
    return tensor_idx

# utility function to generate mini-batch index
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

# to kepler GPU machine still suffer from the memory issue, try to segment into smaller partitions
def part_data(x,y,num_parts):
    # get idx list for the partitions
    num_data = x.shape[1]
    idx = np.arange(num_data)
    parts_idx = np.array_split(idx,num_parts)
    
    # partitioned dataset
    x_ = []
    y_ = []
    
    for i in range(len(parts_idx)):
        x_.append(x[:,parts_idx[i],:])
        y_.append(y[:,parts_idx[i]].transpose())
        
    return x_,y_
    
"""
Test case for training and testing lstm
mask is not needed in this case
"""
def test_lstm(learning_rate=1.,
              decay=2, # learning rate decay
              batch_size = 20,
              seq_length = 20,
              vocab_size = 10000,
              n_hidden = 200,
              init_weight = 0.1,
              max_epoch = 4, # point when the learning rate is adjusted
              max_max_epoch = 13,
              max_grad_norm = 5.): # TODO: need to be adjusted
    
    # instantiate network for the test case
    x = T.tensor3(dtype=dtype)
    # y = T.tensor3(dtype=dtype) 
    #     Note: the output type change back to normal idx other than the one-hot representation
    #          to compute the cost function
    y = T.matrix(dtype='int32')
    lr = T.scalar(dtype=dtype)
        
    # initialize the network 
    lstm_net = lstm(x,y,vocab_size,n_hidden,vocab_size,seq_length,lr,init_weight,max_grad_norm)
    
    # data interface
    train_data = loader('./ptb.train.txt')
    train_x,train_y = make_seq(train_data,seq_length)
    
    # PROC pull out the convert function out of the loop
    #      re-write the convert function in theanjo-gpu implementation
    # the right question is how to tackle the one-hot representation efficiently
    # found out this will lead to out of memory error, need to be fixed
    # train_x = convert_data(train_x,vocab_size).astype(dtype) 
    
    
    num_train = train_x.shape[1] # number of seqs in train data
    
    valid_data = loader('./ptb.valid.txt')
    valid_x,valid_y = make_seq(valid_data,seq_length)

    # num_valid = valid_x.shape[1]
    # num_valid_half = num_valid/2 # auto cell for int/int!

    valid_x = convert_data(valid_x,vocab_size).astype(dtype) 
    # get smaller partitioned data
    """
    to save memory
    """
    num_parts = 3
    valid_x,valid_y = part_data(valid_x,valid_y,num_parts)
    # NOTE: default double type in numpy cannot fit in 6GB memory, need to be fixed
    # theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info() can gets rough estimate of available memory
    # valid_y = valid_y.transpose()
    
    
    # pass over this batch corresponds to the fully sequential processing
    # rewrite from the original torch code
    test_data = loader('./ptb.test.txt')
    test_x,test_y = make_seq(test_data,seq_length)

    # num_test = test_x.shape[1]
    # num_test_half = num_test/2

    test_x = convert_data(test_x,vocab_size).astype(dtype)
    # test_y = test_y.transpose()
    test_x,test_y = part_data(test_x,test_y,num_parts)
    
    start = timeit.default_timer()
    epoch = 0
    step = 0 # a count for each pass of mini-batch data
    epoch_size = np.ceil(train_x.shape[1]/batch_size) # number of steps/mini-batches per epoch
    learning_rate = learning_rate
    # loop over epochs
    
    
    while epoch < max_max_epoch:
        perps = 0 # perplexity for all the mini-batches so far
        
        kf = get_minibatches_idx(num_train,batch_size,shuffle=False)
        minibatch_id = 0 # idx for mini-batch
        
        for _,train_idx in kf: # loop over the mini-batch for a complete pass of epoch
            # one mini-batch counts as a step/get a mini-batch of train data

            # TODO: need to improve this section for speed, redundent computation
            # found tensor.extra_ops provides to_one_hot operator like the lookuptable in torch
            x = train_x[:,train_idx] # get a matrix where each column is a new seq
            x = convert_data(x,vocab_size).astype(dtype) # TODO need a benchmmark for speed and memory efficiency
            # x = train_x[:,train_idx,:] # abort since the train_x is too large to fit in the memory
            y = train_y[:,train_idx].transpose() # transpose into n_seq x n_steps
            
            # y = convert_data(y,vocab_size).astype(dtype)
            
            # y is the delayed sequence of x
            cost = lstm_net.f_grad_shared(x,y) #? prediction to itself for language modeling, CURRENT BUG
            lstm_net.f_update(learning_rate)
            
            perps += cost # average perplexity for the current mini-batch
            minibatch_id += 1
            
            epoch = step / epoch_size
            
            step += 1 # update step counter
            if step % np.round(epoch_size/10) == 10: # update for each 10 steps? refer to the torch code
                # note the gradient computed for the torch is the gradient for the whole seq
                # so here only normalized by number of seqs in the the mini-batch
                print "epoch = " + str(epoch) + \
                      ", lr = " + str(learning_rate) + \
                      ", train prep. = " + str(np.exp(perps / minibatch_id)) + \
                      ", norm_grad. = " + str(lstm_net.f_show_grad(x,y)*20) + \
                      ", since beginning = " + str((timeit.default_timer() - start)/60) + " mins."
                
            if step % epoch_size == 0: # set validation frequency, 1 for debug, 0 for run
                # run validation
                # divide the validation into small fraction to fit into vram
                # x = valid_x
                # y = valid_y
                # valid_perp = lstm_net.f_cost(x,y)
                valid_perp = 0
                
                for i in range(num_parts):
                    valid_perp += lstm_net.f_cost(valid_x[i],valid_y[i])
                    
                valid_perp /= num_parts
                    
                """
                x1 = valid_x[:,0:num_valid_half,:]
                y1 = valid_y[:,0:num_valid_half].transpose()
                
                valid_perp = lstm_net.f_cost(x1,y1)
                
                x2 = valid_x[:,num_valid_half:-1,:]
                y2 = valid_y[:,num_valid_half:-1].transpose()

                valid_perp += lstm_net.f_cost(x2,y2)
                valid_perp /= 2
                """
                                  
                # the reason for running out of memory is that size(x)~2.9GB,size(output from x) ~ 2.9GB
                # hence it run out of memory
                print "Validation set perplexity: ", str(np.exp(valid_perp))
                
                if epoch > max_epoch:
                    learning_rate /= decay # adjust learning rate
              
    
    print("Training is over.")
    
    # run test
    # x = test_x
    # y = test_y
    # test_perp = lstm_net.f_cost(x,y)
    test_perp = 0
    
    for i in range(num_parts):
        test_perp += lstm_net.f_cost(test_x[i],test_y[i])
        
    test_perp /= num_parts
    
    """
    x1 = test_x[:,0:num_test_half,:]
    y1 = test_y[:,0:num_test_half].transpose()
                
    test_perp = lstm_net.f_cost(x1,y1)
                
    x2 = test_x[:,num_test_half:-1,:]
    y2 = test_y[:,num_test_half:-1].transpose()

    test_perp += lstm_net.f_cost(x2,y2)
    test_perp /= 2
    """
    
    print "Test set perplexity: ", str(np.exp(test_perp))
    

if __name__ == '__main__':
    test_lstm()
