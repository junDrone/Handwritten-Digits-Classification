import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0 / (1.0 + np.exp(-1.0 * np.array(z)))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Create training and test matrices
    train_data = np.zeros([0, 784])
    train_label = []
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.zeros([0, 784])
    test_label = []

    for i in range(10):
        train_chunk = mat.get('train' + str(i))
        test_chunk = mat.get('test' + str(i))

        train_data = np.concatenate((train_data, train_chunk))
        test_data = np.concatenate((test_data, test_chunk))

        #train_label = np.concatenate(
        #    (train_label, np.full((train_chunk.shape[0], 1), i, dtype=np.int)))
        train_label=np.concatenate((train_label,np.ones(train_chunk.shape[0])*i),0);
        #test_label = np.concatenate(
        #     (test_label, np.full((test_chunk.shape[0], 1), i, dtype=np.int)))
        test_label = np.concatenate((test_label,np.ones(test_chunk.shape[0])*i),0)


    # Normalize
    train_data = np.double(train_data) / 255.0
    test_data = np.double(test_data) / 255.0

    # Feature selection
    # Remove features which have the same values for all training examples
    is_feature_useless = np.all(train_data == train_data[0, :], axis=0)
    indices_to_delete = np.nonzero(is_feature_useless)
    train_data = np.delete(train_data, indices_to_delete, axis=1)
    test_data = np.delete(test_data, indices_to_delete, axis=1)

    # Random permutations to split data into training and validation data
    indices = np.random.permutation(train_data.shape[0])

    training_idx, validation_idx = indices[:50000], indices[50000:]
    train_data, validation_data = train_data[
        training_idx, :], train_data[validation_idx, :]
    train_label, validation_label = train_label[
        training_idx, ], train_label[validation_idx, ]

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # one-of-k encoding
    training_label = np.array(training_label)
    number_of_samples = training_label.shape[0]
    index = np.arange(number_of_samples, dtype="int")
    one_of_k_label = np.zeros((number_of_samples, 10))
    one_of_k_label[index, training_label.astype(int)] = 1
    training_label = one_of_k_label

    #Feedforward propogation starts here
    # Adding bias for input layer x
    number_of_samples = training_data.shape[0]
    training_data = np.column_stack(
        (training_data, np.ones(number_of_samples))
        )

    # Compute intermediate layer
    # z=w1.T^x
    z = sigmoid(np.dot(training_data, w1.T))

    # Adding bias for intermediate layer z
    number_of_samples = z.shape[0]
    z = np.column_stack(
        (z, np.ones(number_of_samples))
        )

    # Compute output layer
    # o=w2.T^z
    o = sigmoid(np.dot(z, w2.T))
    #Feedforward propogation ends here

    #Backpropogation starts here
    error_output_layer = (training_label - o) * (1 - o) * o #equation (9)

    grad_w2 = np.dot((-1 * error_output_layer).T, z) #equation (8)

    grad_w1 = np.dot(
        (-1 * (1 - z) * z * (
            np.dot(error_output_layer, w2)
            )
        ).T, training_data
        )  #equation (12)

    grad_w1 = np.delete(grad_w1, n_hidden,0)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    obj_grad = obj_grad / training_data.shape[0]
    #Backpropogation ends here

    #Regularization starts here
    obj_val = np.sum(
        ((training_label - o) * (training_label - o)) / 2) / training_data.shape[0] #equation (6)
    obj_val_reg = (lambdaval / (2 * training_data.shape[0])) * \
        (np.sum(np.square(w1)) + np.sum(np.square(w2)))  #equation (15)
    obj_val = obj_val + obj_val_reg #equation (15)
    #Regularization ends here



    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image

    % Output: 
    % label: a column vector of predicted labels"""

    # Adding bias for input layer x
    number_of_samples = data.shape[0]
    data = np.column_stack((data, np.ones(number_of_samples)))

    # Compute intermediate layer
    # z=w1.T^x
    z = sigmoid(np.dot(data, w1.T))

    # Adding bias for intermediate layer z
    number_of_samples = z.shape[0]
    z = np.column_stack((z, np.ones(number_of_samples)))

    # Compute output layer
    # o=w2.T^z
    o = sigmoid(np.dot(z, w2.T))

    # Find the unit with max probability

    labels = np.argmax(o, axis=1)


    return labels


"""**************Neural Network Script Starts here********************************"""

start = time.clock()
print ("running...")

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate(
    (initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0.1


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize
# module. Check documentation for a working example

opts = {'maxiter': 50, 'disp': True}    # Preferred value.

print ("minimizing error...")
nn_params = minimize(nnObjFunction, initialWeights, jac=True,
                     args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1))                 :].reshape((n_class, (n_hidden + 1)))


# Test the computed parameters

print ("predicting...")
predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Lambda '+str(lambdaval))
print('\n Number of hidden units '+str(n_hidden))
print('\n Training set Accuracy:' +
      str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 *
                                          np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Testing Dataset

print('\n Test set Accuracy:' +
    str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


print ("done...")
print (str(time.clock() - start) + "seconds elapsed...")
pickle.dump((n_hidden,w1,w2,lambdaval),open('params.pickle','wb'))
