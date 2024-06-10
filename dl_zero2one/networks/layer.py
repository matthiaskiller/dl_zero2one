import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return outputs: Outputs, of the same shape as x
        :return cache: Cache, stored for backward computation, of the same shape as x
        """
        shape = x.shape
        outputs, cache= np.zeros(shape), np.zeros(shape)

        outputs = 1 / (1 + np.exp(-x))
        cache = outputs


        return outputs, cache

    def backward(self, dout, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """

        y = cache
        sigmoid_derivative = y * (1-y)
        dx = dout * sigmoid_derivative

        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Outputs, of the same shape as x
        :return cache: Cache, stored for backward computation, of the same shape as x
        """

        outputs = np.maximum(0, x)
        cache = x
        
        return outputs, cache

    def backward(self, dout, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        x = cache
        dx = np.greater(0, x) * dout

        return dx
    
class LeakyRelu:
    def __init__(self, slope=0.01):
        self.slope = slope

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        
        outputs = np.where(x < 0, self.slope*x, x)
        cache = x

       
        return outputs, cache

    def backward(self, dout, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        

        x = cache
        dx = np.where(x < 0, self.slope, 1) * dout

        
        return dx



class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        
        outputs = np.tanh(x)

        cache = outputs

       
        return outputs, cache

    def backward(self, dout, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """

        tanh = cache

        dx = dout * (1 - tanh*tanh)

        return dx

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)
    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    N, M = x.shape[0], b.shape[0]
    out = np.zeros((N,M))

    out = x.reshape(N, -1) @ w + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,
    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    
    dx = (dout @ w.T).reshape(x.shape)

    x_reshaped = np.reshape(x, (N, D))
    
    dw = x_reshaped.T @ dout
    
    db = np.mean(dout, axis=0)
    db = np.reshape(db, b.shape[0])

    dw = x_reshaped.T @  dout
    dw = np.divide(dw, N)

    return dx, dw, db