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