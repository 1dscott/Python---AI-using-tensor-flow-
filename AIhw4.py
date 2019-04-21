# AI 

# CNN tensorflow using street number dataset
# Sample in this dataset has 10 digits (0-9)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime

# COMMENTS
# convert target vector into N x K indicator matrix
# K is 10, which is the number class
#
def y2indicator(y):
    """

    :param y: rank-one target array of size N
    :return: N x K indicator matrix for target
    """
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N,K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

# COMMENTS
# given prediction vector p and target vector t, we will calculate error rate using np,mean
#
def error_rate(t,p):
    return np.mean(t != p)

# COMMENTS !!!!!!!!!!!!!!!!!!!!
# implement convpool function
# parameter of this function include X, which is the input image
# W and b is weight and bias of convlution layer
#
def convpool(X, W, b):
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)



# implement init_filter function. Parameter is shape of filter and poolsize
# shape of filter is (filter_width, filter_hieght, old_num_feature_maps, num_feature_maps)
#
def init_filter(shape, poolsz):
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
    return w.astype(np.float32)

#rearrange function will rearrange the dimensionality of input matrix X
#
def rearrange(X):
    # input is (32, 32, 3, N)
    # output is (N, 32, 32, 3) For TF, color comes last
    return (X.transpose(3, 0, 1, 2) / 255).astype(np.float32)

def main():
    train = loadmat('./large_files/train_32x32.mat') # N = 73257
    test = loadmat('./large_files/test_32x32.mat')  #N = 26032


    # Need to scale! don't leave as 0..255
    # Y is a N x 1 matrix with values 1..10 (MATLAB indexes by 1)
    # So flatten it and make it 0..9
    # Also need indicator matrix for cost calcuation
    Xtrain = rearrange(train['X'])
    # Ytrain is rank one array
    Ytrain = train['y'].flatten() - 1
    print("size of Ytrain", len(Ytrain))
    del train
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain_ind = y2indicator(Ytrain)

    Xtest = rearrange(test['X'])
    Ytest = test['y'].flatten() - 1
    del test
    Ytest_ind = y2indicator(Ytest)


    # gradient descent params
    max_iter = 6
    print_period = 10
    N = Xtrain.shape[0]
    batch_sz = 500
    n_batches = N // batch_sz

    #
    # get number of Xtrain, which is multiple of batch_sz
    # you could also just do N = N / batch_sz*batch_sz
    #
    Xtrain = Xtrain[:73000] #remove comma
    Ytrain = Ytrain[:73000]
    Ytrain_ind = Ytrain_ind[:73000] # This is different from original code
    Xtest = Xtest[:26000] # remove comma
    Ytest = Ytest[:26000]
    Ytest_ind = Ytrain_ind[:26000]
    print()
    # Xtest.shape(26000, 32, 32, 3)
    # Ytest.shape(26000, )
    print('Xtest.shape', Xtest.shape)
    print('Ytest.shape', Ytest.shape)
    print()

    # initial weights of fully connected MLP
    # we only have one hidden layer
    # M is the number of units in the hidden layer and K is number of class
    M = 500
    K = 10

    # poolsz is the shape of pool
    poolsz =(2,2)
    #
    # This is the shpae of filter for first convpool layer
    # (filter_width, filter_heigh, num_color_channels, num_feature_maps)
    W1_shape = (5, 5, 3, 20)

    # COMMENTS
    # call init_filter to get initial value of weight (W1_init)
    #
    W1_init = init_filter(W1_shape, poolsz)

    # COMMENTS
    # initialize the bias (b1_init) using np.zeros.
    # Since we need one bias per each output feature map, the number biase is W1_shape[-1]
    # dtype=np.float32
    b1_init = np.zeros(W1_shape[-1], dtype=np.float32)

    # shape of W2 for second convpool layer
    W2_shape = (5, 5, 20, 50)

    # COMMENTS
    # similarly initialize W2_init and b2_init
    #
    W2_init = init_filter(W2_shape, poolsz)
    b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

    # COMMENTS !!!!!!!!!!!!!!!!
    # define initial value of first hidden layer weight (W3_init) using np.random.randn
    # make sure you normalize the weight using input size and output size of the hidden layer
    # input size of first hidden layer weight is W2_shape[-1]*8*8
    # output size of first hidden layer weight is M

    # why it is 8 * 8
    # (first layer) 32 x 32  16 x 16   (second layer) 16 x 16 8 x 8
    #
    W3_init = np.random.randn(W2_shape[-1]*8*8, M) / np.sqrt((W2_shape[-1]*8*8 + M))



    # COMMENTS
    # define initial value of first hidden layer bias (b3_init) using np.zeros
    # size of b3_init is M
    # dtype = np.float32
    b3_init = np.zeros(M, dtype=np.float32)


    # COMMENTS !!!!!!!!!!!!!!!
    # define weight of output layer (W4_init) and bias (b4_init) output layer
    #
    W4_init = np.random.randn(M, K) / np.sqrt((M + K))
    b4_init = np.zeros(K, dtype=np.float32)



    # COMMENTS
    # define X as tf.placeholder. Its data type is tf.float32 its shape is (None, 32, 32, 3)
    # and its name is 'X'
    #
    tfX = tf.placeholder(tf.float32, shape=(batch_sz, 32, 32, 3), name='X')

    # COMMENTS
    # define T as tf.placeholder. Its data type is tf. float32 its shape is (None, K)
    # and its name is 'T'
    #
    tfT = tf.placeholder(tf.float32, shape=(batch_sz, K), name='T')

    # COMMENTS
    # define tf.Variable W1 and b1 using W1_init.astype(np.float32) and b1_init.astype(np.float32)
    #
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))

    # COMMENTS
    # similarly please define W2, b2, W3, b3, W4, and b4 as tf.Variable
    #
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    W4 = tf.Variable(W4_init.astype(np.float32))
    b4 = tf.Variable(b4_init.astype(np.float32))

    # COMMENTS
    # define Z1 which is output of first convpool layer using convpool function, X, W1 and b1
    #
    Z1 = convpool(tfX, W1, b1)

    # COMMENTS
    # define Z2 which is output of second convpool layer using Z1, W2 and b2
    #
    Z2 = convpool(Z1, W2, b2)

    # all of these are in the process of building graph
    # we need to reshape Z2 for feeding data into network
    # -1 is used to represent 500, which is batch size
    Z2_shape = Z2.get_shape().as_list()
    # Z2r = tf.reshape(Z2, [-1, np.prod(Z2_shape[1:])])
    Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])

    # COMMENTS
    # define Z3, which is the output of hidden layer. Z3 depend on Z2r, W3 and b3
    #
    # Z3 = convpool(Z2r, W3, b3)
    Z3 = tf.nn.relu( tf.matmul(Z2r, W3) + b3)

    # COMMENTS
    # define Yish which is output of the output layer without softmax
    #
    Yish = tf.matmul(Z3, W4) + b4

    # COMMENTs
    # define cost using tf.reduce_sum and tf.nn.softmax_cross_entropy_with_logits
    #
    cost = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=Yish,
            labels=tfT
        )
    )

    # COMMENTS
    # define train_op using train.RMSPropOptimizer
    # learning_rate = 0.0001
    # decay = 0.99
    # momentum = 0.9
    #
    learning_rate = 0.0001
    decay = 0.99
    mu = 0.9
    train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)


    # COMMENTS
    # define predict_op using tf.argmax and Yish
    # we'll use this to calculate the error rate
    #
    predict_op = tf.argmax(Yish, 1)


    # COMMENTS  !!!!!!!!!!!!!!!!
    # test out tensorflow code
    # getting the shape of output for each layer using one batch of data
    # test shape of output of Z1, Z2, Z2r, Z3, Yish

    t0 = datetime.now()
    costs = []
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        Xbatch_T = Xtest[2 * batch_sz: (2 * batch_sz + batch_sz)]
        Ybatch_T = Ytest[2 * batch_sz: (2 * batch_sz + batch_sz)]

        shape_Z1 = session.run(tf.shape(Z1), feed_dict={tfX: Xbatch_T})
        print("shape_Z1: ", shape_Z1)
        print()

        shape_Z2 = session.run(tf.shape(Z2), feed_dict={tfX: Xbatch_T})
        print("shape_Z2: ", shape_Z2)
        print()

        shape_Z2r = session.run(tf.shape(Z2r), feed_dict={tfX: Xbatch_T})
        print("shape_Z2r: ", shape_Z2r)
        print()

        shape_Z3 = session.run(tf.shape(Z3), feed_dict={tfX: Xbatch_T})
        print("shape_Z3: ", shape_Z3)

        shape_Yish = session.run(tf.shape(Yish), feed_dict={tfX: Xbatch_T})
        print("shape_Yish: ", shape_Yish)


    # end session




    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):

                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]

                if len(Xbatch) == batch_sz:
                    # COMMENTS
                    # run train_op using Xbatch and Ybatch for n_batchs and for max_iter
                    #
                    session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})
                    if j % print_period == 0:
                        # due to RAM limitations we need to have fixed size input
                        # As a result, we have this ugly total cost and prediction computation
                        # we need to compute the cost and prediction batach by batch. finally, we add them together
                        test_cost = 0
                        prediction = np.zeros(len(Xtest))


                        # we need to loop through entire test set and add all together
                        #
                        for k in range(len(Xtest) // batch_sz):
                            Xtestbatch = Xtest[k* batch_sz:(k * batch_sz + batch_sz)]
                            Ytestbatch = Ytest_ind[k* batch_sz:(k * batch_sz + batch_sz)]
                            test_cost += session.run(cost, feed_dict={tfX: Xtestbatch, tfT: Ytestbatch})
                            prediction[k * batch_sz:(k * batch_sz + batch_sz)] = session.run(
                                predict_op, feed_dict={tfX: Xtestbatch})

                        # COMMENTS
                        # calculate error rate using prediciton and Ytest
                        # print out cost and error for the test set
                        #
                        err = error_rate(prediction, Ytest)
                        print("iteration: ", i, j, " cost: ", test_cost, " error: ", err)
                        # print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                        costs.append(test_cost)


    print("Elapsed time:", (datetime.now()-t0))
    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()
