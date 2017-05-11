import tensorflow as tf

ALPHASIZE = 98
CELLSIZE = 512
NLAYERS = 3
SEQLEN = 30

def rnn():
	Xd = tf.placeholder(tf.uint8, [None, None])
	X = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)
	Yd = tf.placeholder(tf.uint8, [None, None])
	Y_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)
	Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS])

	# the RNN model
	cell = tf.nn.rnn.cell.GRUCell(CELLSIZE)
	mcell = tf.nn.rnn_cell.MultiRNNCell([cell] * NLAYERS, state_is_tuple=False)
	Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state=Hin)

	# SVM output layer
	Hf = tf.reshape(Hr, [-1, CELLSIZE])
	Ylogits = layers.linear(Hf, ALPHASIZE)
	Y = svm(Ylogits)
	Yp = tf.argmax(Y, 1)
