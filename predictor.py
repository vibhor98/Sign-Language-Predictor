import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from keras.models import load_model
from scipy import ndimage
import tensorflow as tf
import flask
from tensorflow.python.framework import ops

from pre_processing import get_dataset

X_train, X_test, Y_train, Y_test = get_dataset()
app = flask.Flask(__name__)

plt.imshow(X_train[21])
plt.show()
print ("y = " + str(np.squeeze(Y_train[21])))

print ("X_train" + str(X_train.shape))
print ("Y_train" + str(Y_train.shape))
print ("X_test" + str(X_test.shape))
print ("Y_test" + str(Y_test.shape))

'''
X_train(1649, 64, 64, 3)
Y_train(1649, 10)
X_test(413, 64, 64, 3)
Y_test(413, 10)
'''

def create_placeholder(height, width, channels, num_classes):
    X = tf.placeholder(shape=[None, height, width, channels], dtype='float32')
    Y = tf.placeholder(shape=[None, num_classes], dtype='float32')
    return X, Y


def initialize_parameters():
    W1 = tf.get_variable("W1", [4,4,3,8], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [2,2,8,16], initializer=tf.contrib.layers.xavier_initializer())

    parameters = {"W1": W1, "W2": W2}
    print (parameters)
    return parameters


def forward_prop(X, parameters):
    W1, W2 = parameters["W1"], parameters["W2"]

    # First layer
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')

    # Second layer
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

    P = tf.contrib.layers.flatten(P2)
    Z4 = tf.contrib.layers.fully_connected(P, 10, activation_fn=None)

    return Z4


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost


def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    #np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitioning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.01, num_epochs=20,
            minibatch_size=64, print_cost=True):
    global costs_str
    costs = []
    costs_str = []
    (m, height, width, channels) = X_train.shape
    num_classes = Y_train.shape[1]

    X, Y = create_placeholder(height, width, channels, num_classes)

    parameters = initialize_parameters()
    Z3 = forward_prop(X, parameters)
    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches

            if print_cost == True:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                costs.append(minibatch_cost)
                costs_str.append(str(minibatch_cost))

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


# Tests on already trained ResNet50 model.
def testOnResNet(X_test, Y_test):
    model = load_model('ResNet50.h5')
    preds = model.evaluate(X_test, Y_test)
    print ('Loss = ' + str(preds[0]))
    print ('Test Accuracy = ' + str(preds[1]))


# REST API for the model to get the results in JSON format.
@app.route('/predict', methods=['GET'])
def predict():
    data = {'success': True}
    data['test_accuracy'] = str(test_acc)
    data['train_accuracy'] = str(train_acc)
    data['Costs'] = costs_str
    return flask.jsonify(data)


if __name__ == '__main__':
    global train_acc, test_acc, parameters
    train_acc, test_acc, parameters = model(X_train, Y_train, X_test, Y_test)
    #testOnResNet(X_test, Y_test)
    #app.run()
