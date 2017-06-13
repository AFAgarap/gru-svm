import numpy as np
import tensorflow as tf

ALPHASIZE = 98
BATCH_SIZE = 100	# no. of training examples to use per training step
CELLSIZE = 512
NLAYERS = 3	# number of hidden layers
NUM_EPOCHS = 10
SEQLEN = 24
SVMC = 1	# cost function constant for SVM
TRAIN_DATA_FILENAME = 'kyoto2013.txt'

def extract_data(filename):
	# methods to be defined later
	return features, labels

def rnn():
	features, labels = extract_data(TRAIN_DATA_FILENAME)
	
	Xd = tf.placeholder(tf.uint8, [None, None])
	X = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)
	Yd = tf.placeholder(tf.uint8, [None, None])
	Y_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)
	Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS])

	# the RNN model
	cell = tf.contrib.rnn.GRUCell(CELLSIZE)
	mcell = tf.contrib.rnn.MultiRNNCell([cell] * NLAYERS, state_is_tuple=False)
	Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state=Hin)

	# SVM output layer
	Hf = tf.reshape(Hr, [-1, CELLSIZE])
	Ylogits = layers.linear(Hf, ALPHASIZE)
	Y = svm(Ylogits)
	Yp = tf.argmax(Y, 1)
	Yp = tf.reshape(Yp, [BATCH_SIZE, -1])

def svm(Ylogits):
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