import numpy as np
import pprint 



def make_char_dict(chars):
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    return char_to_ix,ix_to_char


def lossFun(inputs, targets, hprev, params,vocab_size):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1]         = np.copy(hprev)
  loss           = 0

  Wxh, Whh = params['Wxh'], params['Whh']
  Why      = params['Why']
  bh, by   = params['bh'], params['by']
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by               # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0])          # softmax (cross-entropy loss)

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby         = np.zeros_like(bh), np.zeros_like(by)
  dhnext           = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy              = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy           += np.dot(dy, hs[t].T)
    dby            += dy
    dh              = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw           = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh            += dhraw
    dWxh           += np.dot(dhraw, xs[t].T)
    dWhh           += np.dot(dhraw, hs[t-1].T)
    dhnext          = np.dot(Whh.T, dhraw)

  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n, params, vocab_size):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x          = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes       = []
  Wxh, Whh   = params['Wxh'], params['Whh']
  Why        = params['Why']
  bh, by     = params['bh'], params['by']

  for t in range(n):
    h     = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y     = np.dot(Why, h) + by
    p     = np.exp(y) / np.sum(np.exp(y))
    ix    = np.random.choice(range(vocab_size), p=p.ravel())
    x     = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

def sample_names(h, seed_ix, n_names, params, vocab_size,char_to_ix):
  """ 
  Get n_names dino names 
  Sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x          = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes       = []
  Wxh, Whh   = params['Wxh'], params['Whh']
  Why        = params['Why']
  bh, by     = params['bh'], params['by']

  for t in range(n_names):
    ix = -100
    while ix != char_to_ix['\n']: 
        h     = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y     = np.dot(Why, h) + by
        p     = np.exp(y) / np.sum(np.exp(y))
        ix    = np.random.choice(range(vocab_size), p=p.ravel())
        x     = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
  return ixes

def runmodel(hyparams,vocab_size,data,chars):
    hidden_size   = hyparams['hidden_size']    # size of hidden layer of neurons
    seq_length    = hyparams['seq_length']     # number of steps to unroll the RNN for
    learning_rate = hyparams['learning_rate']
    n_iterations  = hyparams['n_iterations']

    #Char to Index, Index to Char Dictionary
    char_to_ix, ix_to_char = make_char_dict(chars)

    pp = pprint.PrettyPrinter(indent=4)
    print("Hyper Parameters\n",15*'--+')
    pp.pprint(hyparams)
    #For plots
    maxgrad_arr = np.zeros(n_iterations)
    mingrad_arr = np.zeros(n_iterations)
    loss_arr    = np.zeros(n_iterations)

    # model parameters
    Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
    Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
    bh  = np.zeros((hidden_size, 1)) # hidden bias
    by  = np.zeros((vocab_size, 1)) # output bias

    n, p = 0, 0
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby         = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
    smooth_loss      = -np.log(1.0/vocab_size)*seq_length    # loss at iteration 0

    params_dt =  {}
    while n<n_iterations:

      # prepare inputs (we're sweeping from left to right in steps seq_length long)
      if p+seq_length+1 >= len(data) or n == 0: 
        hprev = np.zeros((hidden_size,1))      # reset RNN memory
        p     = 0                              # go from start of data
      inputs  = [char_to_ix[ch] for ch in data[p  :p+seq_length  ]]
      targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

      
      params_dt['Wxh'], params_dt['Whh']= Wxh, Whh
      params_dt['Why']                  = Why     
      params_dt['bh'], params_dt['by']  = bh, by  


      # sample from the model now and then
      if n % 1000 == 0 and (n_iterations-n)<=2000:
        sample_ix = sample_names(hprev, inputs[0], 5,params_dt,vocab_size,char_to_ix)
        #sample_ix = sample(hprev, inputs[0], 200,params_dt,vocab_size)

        txt       = ''.join(ix_to_char[ix] for ix in sample_ix)
        print ('----\n %s \n----' % (txt, ))

      # forward seq_length characters through the net and fetch gradient
      loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev,params_dt,vocab_size)
      smooth_loss = smooth_loss * 0.9999 + loss * 0.0001
      #smooth_loss = loss
      if n % 1000 == 0 and (n_iterations-n)<=2000: 
        print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
      
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                    [dWxh, dWhh, dWhy, dbh, dby], 
                                    [mWxh, mWhh, mWhy, mbh, mby]):
        mem   +=  dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      #-------------------------------------------------------#
      loss_arr[n]     = loss
      mingrad_arr[n]  = np.min([dWxh,])
      maxgrad_arr[n]  = np.max([dWxh,])
      #-------------------------------------------------------#
      p += seq_length # move data pointer
      n += 1          # iteration counter 


    return (loss_arr,mingrad_arr,maxgrad_arr)
