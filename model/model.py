import numpy as np
import tensorflow as tf

def main():

	# the GRU model
	cell = tf.contrib.rnn.GRUCell(CELLSIZE)

	mcell = tf.contrib.rnn.MultiRNNCell([cell] * NLAYERS, state_is_tuple=False)

	Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state=Hin)

	# the SVM classifier
	
	# Weights that inform how each feature affect the classification
	W = tf.Variable(tf.zeros([num_features, 1]))
	b = tf.Variable(tf.zeros([1]))
	y_raw = tf.matmul(x, W) + b

	# Optimization
	regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
	hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE, 1]), 1 - y * y_raw))
	svm_loss = regularization_loss + svmC * hinge_loss;
	train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(svm_loss)

	# Evaluation
	predicted_class = tf.sign(y_raw)
	correct_prediction = tf.equal(y, predicted_class)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

	# Create a local session to run this computation
	with tf.Session() as sess:

		tf.initialize_all_variables().run()

		# Iterate and train
		for step in range(num_epochs * train_size // BATCH_SIZE):