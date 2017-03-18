import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    f = X[i,:].dot(W)
    f -= np.max(f)
    sigma = np.sum(np.exp(f))
    loss += -f[y[i]]+np.log(sigma)

    dW += np.reshape(X[i,:],(-1,1)).dot(np.reshape(np.exp(f),(1,-1)))/sigma
    dW[:,y[i]] -= X[i,:]

  loss /= X.shape[0]
  loss += 0.5*reg*np.sum(W**2)
  dW /= X.shape[0]
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  f -= np.reshape(np.max(f,axis=1),(-1,1))
  z = np.exp(f)
  loss = np.sum(-f[np.arange(X.shape[0]),y]+np.log(np.sum(z,axis=1)),axis=0)/X.shape[0] + 0.5*reg*np.sum(W**2)

  y_matrix = np.zeros_like(f)
  y_matrix[np.arange(len(y)),y] = 1

  dW = (np.transpose(X).dot(z/np.reshape(np.sum(z,axis=1),(-1,1))-y_matrix))/X.shape[0] + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

