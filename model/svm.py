import numpy as np
import scipy.io as io
import tensorflow as tf

BATCH_SIZE = 100    # The number of training examples to use per training step.
NUM_EPOCHS = 10     # The number of epochs for training
TRAIN_DATA_FILENAME = 'sample.csv'
SVMC = 1

def extract_data(filename):

    out = np.loadtxt(filename, delimiter=',');

    # Arrays to hold the labels and feature vectors.
    labels = out[:,0]
    labels = labels.reshape(labels.size, 1)
    feature = out[:,1:]

    # Return a pair of the feature matrix and the one-hot label matrix.
    return feature, labels


def main():
    # Extract data into NumPy matrices
    train_data, train_labels = extract_data(train_data_filename)

    # y \in {-1, 1}
    train_labels[train_labels == 0] = -1

    # Shape of the training data.
    train_size, num_features = train_data.shape

    # Number of epochs for training
    num_epochs = NUM_EPOCHS

    # Get the C param of SVM
    svmC = SVMC

    # Training samples and labels are to be fed into the graph
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y = tf.placeholder("float", shape=[None, 1])

    # Define and initialize the network.

    # These are the weights that inform how much each feature contributes to
    # the classification.
    W = tf.Variable(tf.zeros([num_features, 1]))
    b = tf.Variable(tf.zeros([1]))
    y_raw = tf.matmul(x, W) + b

    # Optimization.
    regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE, 1]), 1 - y * y_raw));
    svm_loss = regularization_loss + svmC * hinge_loss;
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)

    # Evaluation.
    predicted_class = tf.sign(y_raw);
    correct_prediction = tf.equal(y,predicted_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create a local session to run this computation.
    with tf.Session() as s:

        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()

        # Iterate and train.
        for step in range(num_epochs * train_size // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset : (offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            train_step.run(feed_dict={x : batch_data, y : batch_labels})
            print('loss: ', svm_loss.eval(feed_dict={x : batch_data, y : batch_labels}))

            if offset >= train_size-BATCH_SIZE:
                print()

            print()
            print('Weight matrix.')
            print(s.run(W),'\n')
            print('Bias vector.')
            print(s.run(b),'\n')
            print("Applying model to first test instance.\n")

        print("Accuracy on train:", accuracy.eval(feed_dict={x: train_data, y: train_labels}))

if __name__ == '__main__':
    tf.app.run()
