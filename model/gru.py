import numpy as np
import tensorflow as tf

class GRU(object):

	def __init__(self, input_size, output_size, hidden_size):
		'''Class implementing GRU'''

		# Input weights
		self.Wxh = np.random.randn(hidden_size, input_size) * 1
		self.Wxr = np.random.randn(hidden_size, input_size) * 1
		self.Wxz = np.random.randn(hidden_size, input_size) * 1

		# Recurrent weights
		self.Rhh = np.random.randn(hidden_size, hidden_size) * 1
		self.Rhr = np.random.randn(hidden_size, hidden_size) * 1
		self.Rhz = np.random.randn(hidden_size, hidden_size) * 1

		# Biases
		self.bh = np.zeros((hidden_size, 1))
		self.br = np.zeros((hidden_size, 1))
		self.bz = np.zeros((hidden_size, 1))

		# Weight from hidden layer to output layer
		self.Why = np.random.randn(output_size, hidden_size) * 1

		self.weights = [self.Wxh, self.Wxr, self.Wxz, self.Rhh, self.Rhr, self.Rhz,
						self.bh, self.br, self.bz, self.Why]

	def sigmoid(self, x):
		''' The sigmoid activation function '''
		sigmoid_val= 1 / (1 + np.exp(-x))
		return sigmoid_val

	def update_gate(self, Wxz, xs, Rhz, hs, bz):
		''' Update Gate for GRU '''
		z_bar = np.dot(Wxz, xs) + np.dot(Rhz, hs) + bz
		z = sigmoid(z_bar)
		return z

	def reset_gate(self, Wxr, xs, Rhr, hs, br):
		''' Reset Gate for GRU '''
		r_bar = np.dot(Wxr, xs) + np.dot(Rhr, hs) + br
		r = sigmoid(r_bar)
		return r

	def hidden_cell_bar(self, Wxh, xs, Rhh, rs, hs, bh):
		''' Candidate Value for Cell Output '''
		h_bar = np.dot(Wxh, xs) + np.dot(Rhh, np.multiply(rs, hs)) + bh
		h = np.tanh(h_bar)
		return h

	def hidden_cell(self, hs, zs, hs_prev):
		'''
		Compute new cell output by interpolating
		between candidate value and old
		'''
		ones = np.ones_like(zs)
		h = np.multiply(hs, zs) + np.multiply(hs_prev, ones - zs)
		return h