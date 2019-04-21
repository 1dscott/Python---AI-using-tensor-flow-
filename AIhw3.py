# AI

# Vanilla deep network using tensor flow
# The dataset is street view house number
# Street view house number has 10 digits
# float32

# dataset link: http://ufldl.stanford.edu/housenumbers
# sataset you need is Format2: Copped Digits: train_32x32.mat, test_32x32.mat

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime



# comments
# convert y unto indicator matrix
# N is the size of dataset
# K is the number of class
def y2indicator(y):
    """

    :param y: rank-one target array of size N
    :return: N * K indicator matrix for target
    """
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N,K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

# comments
# calculate the error rate
# p is the prediction vector
# t is the target vector
def error_rate(p, t):
    return np.mean(t != p)


def flatten(X):
    """

    :param X: input will be (32,32,3,N) basically input is 4d matrix
    :return: output will be (N, 3072) 2d numpy matrix
    """
    N = X.shape[-1] #matlab has num of rows as last deimension
    flat = np.zeros((N, 3072))  # 3074 = 32 x 32 x 3
    for i in range(N):
        flat[i] = X[:, :, :, i].reshape(3072)
    return flat

def forward(self, X):
    """

    :param X: intermediate tensorflow expression
    :return: output expression for hidden layer
    """
    return tf.nn.relu(tf.matmul(X, W) + b)


def main():

    # reading the train and test data
    # we do not need split data into train and test since it is already done for us
    train = loadmat('./large_files/train_32x32.mat')
    test = loadmat('./large_files/test_32x32.mat')

    print('type of train: ', type(train))
    print('type of train[x]', type(train['X']))

    # Need to scale training set! dont leave as 0..255
    # Y is a N x 1 atrix with values 1..10(MATLAB indexes by 1)
    # So flatten it and make it 0..9
    # Also need indicator matrix for cost calulation
    # flatten means combining several dimensions into one dimension

    # each element of 4d vector is real number
    Xtrain = flatten(train['X'].astype(np.float32) / 255)

    # Ytrain is rank-one array
    Ytrain = train['y'].flatten() - 1 # subtract one from all elements

    print('shape of train[Y] before flatten: ', train['y'].shape)
    print('shape of Ytrain after flatten: ', Ytrain.shape)

    # we shuffle it so that we can get different results each time
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain_ind = y2indicator(Ytrain)

    Xtest = flatten(test['X'].astype(np.float32)/255)
    Ytest = test['y'].flatten() -1  # Ytest is rank-one arrayy
    Ytest_ind = y2indicator(Ytest)

    # comments
    # maximum number of iteration (max_iter) is 20
    # batch size (batch_sz) is 500
    # You need to figure out the num of batches(n_batches)
    max_iter = 20
    batch_sz = 500

    # comments
    # we will use two hidden layer architecture
    # First hidden layer M1 has 1000 hidden unit
    # second hidden layer M2 has 500 hidden unit
    # number of class K is 10
    # use np.random.randn to initialize the W1_init, W2_init and W3_init
    # use np.zeros to initialize b1_init, b2_init, b3_init
    #
    # W1_init is used to initialize the value of tensorflow variable W1
    #  similarly define tf.Variable including W2, b2, W3, and b3


    M1 = 1000
    M2 = 500
    K = 10
    N, D = Xtrain.shape
    n_batches = N // batch_sz
    W1_init= np.random.randn(D, M1) / np.sqrt(D)
    W2_init= np.random.randn(M1, M2) / np.sqrt(M1)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b1_init = np.zeros((1, M1))
    b2_init = np.zeros((1, M2))
    b3_init = np.zeros((1, K))

    # define W1 as tf.Variable and use W1_init.astype(np.float32) as its initial value
    W1 = tf.Variable(W1_init.astype(np.float32))

    # define b1 as tf.Variable and use b1_init.astype(np.float32) as its initial value
    b1 = tf.Variable(b1_init.astype(np.float32))

    # similarly define tf.Variable including W2, b2, W3 and b3
    #
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    # comments
    # everything in tesnor flow is float32
    # define X as tf.placeholder with datatype tf.float32 and its shape is (None, D)
    # X is input data
    tfX = tf.placeholder(tf.float32, shape=(None, D), name='x')

    # define T as tf.placeholder with datatype tf.float32 and its shape is (None, K)
    # T is the target
    tfT = tf.placeholder(tf.float32, shape=(None, K), name='T')

    # comments
    # use tensor flow function to define A1, which is the first hidden layer output
    # define A2, which is the second hidden layer output
    # for A2, and A3, use tf.nn.relu as the activation fnction
    # define z3, which is the output of the output layer before passing through softmax
    # remember, the cost function does the softmaxing
    #
    A1 = tf.nn.relu(tf.matmul(tfX, W1) + b1)
    A2 = tf.nn.relu(tf.matmul(A1, W2) + b2)
    Z3 = tf.matmul(A2, W3) + b3

    # Define cost using tf.reduce_sum and tf.nn.softmax_cross_entropy_with_logits
    #

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=Z3,
            labels=tfT
        )
    )

    # define train_op using tf.train.RMSPropOptimizer
    # learning rate is 0.0001
    # decay - 0.99
    # momentum = 0.9
    # we choose the optimizer but don't implement the alorithm ourselves
    #
    learning_rate = 0.0001             # learning rate is 0.0001
    decay = 0.99             # decay is .99
    mu = 0.9                 # momentum -or- mu is .9
    train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)

    # Define prediction to produce the prediction results
    # Use tf.argmax(Z3,1)
    predict = tf.argmax(Z3, 1)

    costs = []
    # initialize all variables
    init = tf.global_variables_initializer()

    # Use one batch to check the deimensionality of output value for hidden layers and loutput later
    with tf.Session() as session:
        session.run(init)
        batch_no = 2
        Xbatch_T = Xtrain[batch_no * batch_sz:(batch_no * batch_sz + batch_sz)]
        Ybatch_T = Ytrain_ind[batch_no * batch_sz:(batch_no * batch_sz + batch_sz)]

        shape_A1 = session.run(tf.shape(A1), feed_dict={tfX: Xbatch_T})
        print("shape of A1: ", shape_A1)
        print()

        shape_act = session.run(tf.shape(Z3), feed_dict={tfX: Xbatch_T})
        print("shape of act: ", shape_act)

        for i in range(50):
            # run RMSprop
            session.run(train_op, feed_dict={tfX: Xbatch_T, tfT: Ybatch_T})
            value_cost = session.run(cost, feed_dict={tfX: Xbatch_T, tfT: Ybatch_T})
            print("Value of cost after update: ", value_cost)
            print()

    # Use batch gradient descent to train neural networks
    # for every 10 batches, print out test cost and test error using testing set
    # draw the graph for the testing cost
    error = 1
    with tf.Session() as session:
        session.run(init)
        for i in range(max_iter):
            for j in range(n_batches):
                # Xbatch is training data in one batch
                # Ybatch is target indicator matrix in one batch
                Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz)]
                Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz)]
                # train neural network for one iteration
                session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})

                # print cost and error on validation set at every 10 steps
                if j % 10 == 0:
                    c = session.run(cost, feed_dict={tfX: Xtest, tfT: Ytest_ind})
                    costs.append(c)

                    # produce prediction for validation set
                    p = session.run(predict, feed_dict={tfX: Xtest, tfT: Ytest_ind})
                    e = error_rate(np.argmax(Ytest_ind, 1),p)
                    print("i:", i, "j: ", j, "cost: ", c, "error rate: ", e)
                    if error > e:
                        error = e
    print("Final errorr rate: ", error)
    print("Final accuracry: ", 1 - error)

    print()
    plt.plot(costs)
    plt.show()



if __name__ == '__main__':
    main()
