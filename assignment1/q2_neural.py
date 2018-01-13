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

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    #print data.shape, params.shape, Dx, H, Dy
    

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation

    z1 = np.dot(data,W1) + b1
    h = sigmoid(z1)

    z2 = np.dot(h,W2) + b2
    y_hat = softmax(z2)

    cost = -np.sum(np.multiply(labels, np.log(y_hat)))

    ### END YOUR CODE
    
    res = y_hat - labels
    ### YOUR CODE HERE: backward propagation
    gradW2 = np.dot(h.T, res)
    gradb2 = np.sum(res, axis = 0)


    dh = np.dot(res, W2.T)
    dz1 = sigmoid_grad(h) * dh
    
    gradW1 = np.dot(data.T, dz1)
    gradb1 = np.sum(dz1, axis = 0)

    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    

    # Checking if the weight vectors and gradients are of the same size
    # print gradW2.shape, W2.shape
    # print gradW1.shape, W1.shape
    # print b1.shape
    # print b2.shape
    # print grad.shape
    return cost, grad

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


    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    N = 40
    dimensions = [20, 10, 20]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )


    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()