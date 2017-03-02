import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    num_train = data.shape[0]
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])


    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))




    #Forward Feeding - calculating the loss
    h = sigmoid(np.dot(data, W1) + b1)
    
    score = softmax(np.dot(h, W2) + b2)
    loss = - np.sum(np.log(score[xrange(num_train), np.argmax(labels, axis = 1)]))

    # Calculating gradient by taking care of gradient matrices in the hidden
    # layers
    score -= labels

    grad_layer_2   = np.zeros_like(score)
    grad_layer_1 = np.zeros_like(h)
    gradW2    = np.zeros_like(W2)
    gradW1    = np.zeros_like(W1)
    gradb2    = np.zeros_like(b2)
    gradb1    = np.zeros_like(b1)

    gradW2 = np.dot(h.T, score)
    gradb2 = np.sum(score, axis = 0)
    grad_layer_2 = np.dot(score, W2.T)
    grad_layer_1 = sigmoid_grad(h) * grad_layer_2

    gradW1 = np.dot(data.T, grad_layer_1)
    gradb1 = np.sum(grad_layer_1, axis = 0)

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return loss, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
