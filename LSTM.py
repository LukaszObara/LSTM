"""LSTM.py
~~~~~~~~~~~~~~
A very simple long short term memory neural netowrk based of network.py,
and RecurrentNetwork.py implementing the stochastic gradient descent 
learning algorithm. The network is composed of 1 (one) hidden layer and 
incorporates concepts such as a forget gate, input gate, and output gate
to improve the performance of the RNN. The cross-entropy cost function 
is used the evaluate compare perceived results to desired reuslts. 
"""

# TODO: l_1 Regularization 
# The first term is just the usual expression for the cross-entropy. But 
# we've added a second term, namely the sum of the squares of all the 
# weights in the network.


#### Libraries ####
# Third Party Libraries
import numpy as np 

# User Libraries
from Text_Vec import Text_Vec

__version__ = "0.6.1"
__author__ = "Lukasz Obara"

# Misc functions 
def softmax(z):
	""" The softmax function.
	"""
	return np.exp(z) / np.sum(np.exp(z))

def sigmoid(z):
	""" The sigmoid function used in activating the neurons. 
	"""
	return 1.0/(1.0+np.exp(-z))

def d_sigmoid(z):
	""" The derivative of the sigmoid function, used for gate control.
	"""
	return sigmoid(z)*(1-sigmoid(z))

class LSTM(object):
	def __init__(self, data, hidden_size, eps=0.0001):
		self.data = data
		self.time_periods = len(data)-1
		self.hidden_size = hidden_size
		# initializes the weights for the given gates of an input signal 
		self.weight_W_f = np.random.rand(hidden_size, len(data[0])) * 0.1 # W_f
		self.weight_W_i = np.random.rand(hidden_size, len(data[0])) * 0.1 # W_i
		self.weight_W_g = np.random.rand(hidden_size, len(data[0])) * 0.1 # W_g
		self.weight_W_o = np.random.rand(hidden_size, len(data[0])) * 0.1 # W_o
		# initializes the weights for the given gates for the previous 
		# hidden state.
		self.weight_U_f = np.random.rand(hidden_size, hidden_size) * 0.1 # U_f
		self.weight_U_i = np.random.rand(hidden_size, hidden_size) * 0.1 # U_i
		self.weight_U_g = np.random.rand(hidden_size, hidden_size) * 0.1 # U_g
		self.weight_U_o = np.random.rand(hidden_size, hidden_size) * 0.1 # U_o
		# initializes the biasses for the given gates
		self.bias_f = np.array([np.ones(hidden_size)]).T * 1.5 # b_f
		self.bias_i = np.array([np.ones(hidden_size)]).T * 1.5 # b_i
		self.bias_g = np.array([np.ones(hidden_size)]).T * 1.5 # b_g
		self.bias_o = np.array([np.ones(hidden_size)]).T * 1.5 # b_o

		# initializes the weight and bias for the output layer
		# This last layer is identical to that found in the RNN that was
		# built before. It has the role of setting the correct dimension
		# before being subjected to the loss function. 
		self.weight_V = np.random.rand(len(data[0]), hidden_size) * 0.1 # V
		self.bias_d = np.array([np.ones(len(data[0]))]).T * 1.5 # d
		# Initialization of hidden state for time 0.
		self.h_0 = np.array([np.zeros(hidden_size)]).T
		self.cell_0 = np.array([np.zeros(self.hidden_size)]).T

		# Initial cache values for update of RMSProp
		self.cache_W_f = np.zeros((hidden_size, len(data[0])))
		self.cache_W_i = np.zeros((hidden_size, len(data[0])))
		self.cache_W_g = np.zeros((hidden_size, len(data[0])))
		self.cache_W_o = np.zeros((hidden_size, len(data[0])))
		self.cache_U_f = np.zeros((hidden_size, hidden_size))
		self.cache_U_i = np.zeros((hidden_size, hidden_size))
		self.cache_U_g = np.zeros((hidden_size, hidden_size))
		self.cache_U_o = np.zeros((hidden_size, hidden_size))
		self.cache_bias_f = np.zeros((hidden_size, 1))
		self.cache_bias_i = np.zeros((hidden_size, 1))
		self.cache_bias_g = np.zeros((hidden_size, 1))
		self.cache_bias_o = np.zeros((hidden_size, 1))
		self.cache_V = np.zeros((len(data[0]), hidden_size))
		self.cache_bias_d = np.zeros((hidden_size, 1))
		self.eps = eps

	def train(self, seq_length, epochs, eta, decay_rate=0.9, learning_decay=0.0):
		"""Updates the network's weights and biases by applying gradient
		descent using backpropagation through time. 
		"""
		sequences = [self.data[i:i+seq_length] \
			 for i in range(0, len(self.data), seq_length)] 

		for epoch in range(epochs):
			print('epoch {}'.format(epoch))

			for seq in sequences:
				self.update(seq, epoch, eta, decay_rate, learning_decay)
			final_text = ''
			_, _, _, _, _, _, outputs = self.feedforward(seq)
			for j in outputs:
				num = np.argmax(j)
				final_text += chr(num)

			print(final_text)

	def update(self, seq, epoch, eta, decay_rate, learning_decay):
		def update_rule(cache_attr, x_attr, dx):
			cache = getattr(self, cache_attr)
			cache = decay_rate * cache + (1 - decay_rate) * dx**2
			setattr(self, cache_attr, cache)

			x = getattr(self, x_attr)
			x -= eta * dx / (np.sqrt(cache) + self.eps)
			setattr(self, x_attr, x)

		# cache_W_f, cache_W_i, cache_W_g, cache_W_o = 0, 0, 0, 0 
		# cache_U_f, cache_U_i, cache_U_g, cache_U_o = 0, 0, 0, 0
		# cache_bias_f, cache_bias_i, cache_bias_g, cache_bias_o = 0, 0, 0, 0
		cache_V, cache_bias_d = 0, 0 
		eps = 0.0001

		delta_nabla_bias_f, delta_nabla_bias_i, delta_nabla_bias_g,\
		delta_nabla_bias_o, delta_nabla_U_f, delta_nabla_U_i,\
		delta_nabla_U_g, delta_nabla_U_o, delta_nabla_W_f, delta_nabla_W_i,\
		delta_nabla_W_g, delta_nabla_W_o, delta_nabla_bias_d, \
		delta_nabla_V = self.backward_pass(seq)
			
		# Hidden weight change of each gate
		update_rule('cache_W_f', 'weight_W_f', delta_nabla_W_f)
		update_rule('cache_W_i', 'weight_W_i', delta_nabla_W_i)
		update_rule('cache_W_g', 'weight_W_g', delta_nabla_W_g)
		update_rule('cache_W_o', 'weight_W_o', delta_nabla_W_o)

		# Input weight change for each gate
		update_rule('cache_U_f', 'weight_U_f', delta_nabla_U_f)
		update_rule('cache_U_i', 'weight_U_i', delta_nabla_U_i)
		update_rule('cache_U_g', 'weight_U_g', delta_nabla_U_g)
		update_rule('cache_U_o', 'weight_U_o', delta_nabla_U_o)

		# Bias change for each gate
		update_rule('cache_bias_f', 'bias_f', delta_nabla_bias_f)
		update_rule('cache_bias_i', 'bias_i', delta_nabla_bias_i)
		update_rule('cache_bias_g', 'bias_g', delta_nabla_bias_g)
		update_rule('cache_bias_o', 'bias_o', delta_nabla_bias_o)

		# Output change
		update_rule('cache_V', 'weight_V', delta_nabla_V)
		# update_rule('cache_bias_d', 'bias_d', delta_nabla_bias_d)
		
		# # Hiddent weight change of each gate
		# cache_W_f = decay_rate * cache_W_f \
		# 		  + (1 - decay_rate) * delta_nabla_W_f**2
		# self.weight_W_f -= eta * delta_nabla_W_f / (np.sqrt(cache_W_f) + eps)

		# cache_W_i = decay_rate * cache_W_i \
		# 		  + (1 - decay_rate) * delta_nabla_W_i**2
		# self.weight_W_i -= eta * delta_nabla_W_i / (np.sqrt(cache_W_i) + eps)

		# cache_W_g = decay_rate * cache_W_g \
		# 		  + (1 - decay_rate) * delta_nabla_W_g**2
		# self.weight_W_g -= eta * delta_nabla_W_g / (np.sqrt(cache_W_g) + eps)
		
		# cache_W_o = decay_rate * cache_W_o \
		# 		  + (1 - decay_rate) * delta_nabla_W_o**2
		# self.weight_W_o -= eta * delta_nabla_W_o / (np.sqrt(cache_W_o) + eps)

		# # Input weight change for each gate
		# cache_U_f = decay_rate * cache_U_f \
		# 		  + (1 - decay_rate) * delta_nabla_U_f**2
		# self.weight_U_f -= eta * delta_nabla_U_f / (np.sqrt(cache_U_f) + eps)			
		
		# cache_U_i = decay_rate * cache_U_i \
		# 		  + (1 - decay_rate) * delta_nabla_U_i**2
		# self.weight_U_i -= eta * delta_nabla_U_i / (np.sqrt(cache_U_i) + eps)

		# cache_U_g = decay_rate * cache_U_g \
		# 		  + (1 - decay_rate) * delta_nabla_U_g**2
		# self.weight_U_g -= eta * delta_nabla_U_g / (np.sqrt(cache_U_g) + eps)

		# cache_U_o = decay_rate * cache_U_o \
		# 		  + (1 - decay_rate) * delta_nabla_U_o**2
		# self.weight_U_o -= eta * delta_nabla_U_o / (np.sqrt(cache_U_o) + eps)

		# # Change for the biases for each gate
		# cache_bias_f = decay_rate * cache_bias_f \
		# 			 + (1 - decay_rate) * delta_nabla_bias_f**2
		# self.bias_f -= eta * delta_nabla_bias_f / (np.sqrt(cache_bias_f) + eps)

		# cache_bias_i = decay_rate * cache_bias_i \
		# 			 + (1 - decay_rate) * delta_nabla_bias_i**2
		# self.bias_f -= eta * delta_nabla_bias_i / (np.sqrt(cache_bias_i) + eps)

		# cache_bias_g = decay_rate * cache_bias_g \
		# 			 + (1 - decay_rate) * delta_nabla_bias_g**2
		# self.bias_g -= eta * delta_nabla_bias_g / (np.sqrt(cache_bias_g) + eps)

		# cache_bias_o = decay_rate * cache_bias_o \
		# 			 + (1 - decay_rate) * delta_nabla_bias_o**2
		# self.bias_o -= eta * delta_nabla_bias_o / (np.sqrt(cache_bias_o) + eps)

		# # Change for the outputs
		# cache_V += decay_rate * cache_V \
		# 		+ (1 - decay_rate) * delta_nabla_V**2
		# self.weight_V -= eta * delta_nabla_V/ (np.sqrt(cache_V) + eps)

		cache_bias_d += decay_rate * cache_bias_d \
					 + (1 - decay_rate) * delta_nabla_bias_d**2
		self.bias_d -= eta * delta_nabla_bias_d / (np.sqrt(cache_bias_d) + eps)

	def feedforward(self, sequence):
		"""Returns a tuple of lists `(f_t, i_t, g_t, o_t, C_t, h_t, p_t)`
		representing the forget, input, gate, and output gates, the cell,
		the hidden, and output states at each time step. Each element in 
		the tuple is a numpy array.  
		"""
		# Lists for the activations of the gates and states at a given 
		# time step
		f_t, i_t, g_t, o_t, p_t = [], [], [], [], []
		h_t = [self.h_0]
		C_t = [self.cell_0]
		
		for t in range(self.time_periods): 
			# Forget gate
			act_f = np.dot(self.weight_W_f, self.data[t]) \
					+np.dot(self.weight_U_f, h_t[t]) \
					+self.bias_f
			f_gate = sigmoid(act_f)
			f_t.append(f_gate)

			# Input gate
			act_i = np.dot(self.weight_W_i, self.data[t]) \
					+np.dot(self.weight_U_i, h_t[t]) \
					+self.bias_i
			i_gate = sigmoid(act_i)
			i_t.append(i_gate)

			# Gate
			act_g = np.dot(self.weight_W_g, self.data[t]) \
					+np.dot(self.weight_U_g, h_t[t]) \
					+self.bias_g
			gate = np.tanh(act_g)
			g_t.append(gate)

			# Output gate 
			act_o = np.dot(self.weight_W_o, self.data[t]) \
					+np.dot(self.weight_U_o, h_t[t]) \
					+self.bias_o
			o_gate = sigmoid(act_g)
			o_t.append(o_gate)

			# Cell state
			cell_t = np.multiply(f_gate, C_t[t]) + np.multiply(i_gate, gate)
			C_t.append(cell_t)

			# Hidden states
			hidden_t = np.multiply(o_gate, np.tanh(cell_t))
			h_t.append(hidden_t)

			# Output States
			out_t = np.dot(self.weight_V, hidden_t) + self.bias_d
			p_t.append(out_t)

		return (f_t, i_t, g_t, o_t, C_t, h_t, p_t)

	def backward_pass(self, sequence):
		"""Returns a tuple `(nabla_bias_f, nabla_bias_i, nabla_bias_g, 
		nabla_bias_o, nabla_U_f, nabla_U_i, nabla_U_g, nabla_U_o,
		nabla_W_f, nabla_W_i, nabla_W_g, nabla_W_o)` represeting the
		gradient for the cross entropy cost function. Each element in 
		the tuple is a numpy array.  
		"""
		f_states, i_states, g_states, o_states, \
		C_states, h_states, p_states = self.feedforward(sequence)

		# The last time step
		nabla_p = self.d_loss(softmax(p_states[-1]),self.data[-1])
		nabla_d = nabla_p
		nabla_V = np.dot(nabla_p, h_states[-1].T)

		# `h_states` includes `h_0` hence making h_states of length 
		# `len()+1`. As such, we could couple it with the inputs at time
		# `t`

		# `nabla_h` for the last time step.
		nabla_h = np.dot(self.weight_V.T, nabla_p)

		# Preliminary computations needed for computing the change for 
		# the biases, inputs weights, hidden weights, cell states, and
		# output states for the last time step 
		diag_f = np.diag(f_states[-1][:,0])
		diag_i = np.diag(i_states[-1][:,0])
		diag_g = np.diag(g_states[-1][:,0])
		diag_o = np.diag(o_states[-1][:,0])

		nabla_forget = np.dot(diag_f, (np.identity(self.hidden_size)-diag_f))
		nabla_input = np.dot(diag_i, (np.identity(self.hidden_size)-diag_i)) 
		nabla_gate = np.identity(self.hidden_size)-diag_g**2
		nabla_output = np.dot(diag_o, (np.identity(self.hidden_size)-diag_o))

		# nabla_forget = np.multiply(f_states[-1], 1-f_states[-1])
		# nabla_input = np.multiply(diag_i[-1], 1-diag_i[-1])
		# nabla_gate = 1-diag_g[-1]**2
		# nabla_output = np.multiply(diag_o[-1], 1-diag_o[-1])

		# temp = np.tanh(C_states[-1])
		# d_tanh = 1 - temp**2

		# nabla_bias_f = np.multiply(np.multiply(np.multiply(nabla_forget, d_tanh),\
		# 						   C_states[-2]), nabla_h)
		# nabla_bias_i = np.multiply(np.multiply(np.multiply(nabla_input, d_tanh),\
		# 						   g_states[-1]), nabla_h)


		temp = np.diag(np.tanh(C_states[-1])[:,0])
		d_tanh = np.identity(self.hidden_size) - temp**2
		diag_C = np.diag(C_states[-2][:,0])

		nabla_bias_f = np.dot(np.dot(np.dot(np.dot(nabla_forget, d_tanh), diag_C),\
							  diag_o),nabla_h)
		nabla_bias_i = np.dot(np.dot(np.dot(np.dot(nabla_input, d_tanh), diag_g),\
							  diag_o),nabla_h)
		nabla_bias_g = np.dot(np.dot(np.dot(np.dot(nabla_gate, d_tanh), diag_i),\
							  diag_o),nabla_h)
		nabla_bias_o = np.dot(np.dot(nabla_output, temp), nabla_h)
 
		nabla_U_f = np.dot(nabla_bias_f, h_states[-2].T) 
		nabla_U_i = np.dot(nabla_bias_i, h_states[-2].T) 
		nabla_U_g = np.dot(nabla_bias_g, h_states[-2].T) 
		nabla_U_o = np.dot(nabla_bias_o, h_states[-2].T)

		nabla_W_f = np.dot(nabla_bias_f, self.data[-2].T)
		nabla_W_i = np.dot(nabla_bias_i, self.data[-2].T)
		nabla_W_g = np.dot(nabla_bias_g, self.data[-2].T)
		nabla_W_o = np.dot(nabla_bias_o, self.data[-2].T)

		nabla_C = np.dot(np.dot(self.weight_U_f.T, nabla_forget), diag_C)\
				 +np.dot(np.dot(self.weight_U_i.T, nabla_input), diag_g)\
				 +np.dot(np.dot(self.weight_U_g.T, nabla_gate), diag_i)

		for t in reversed(range(1, self.time_periods)):
			# We start by computing `nabla_p` for the second to last 
			# step and apply the backprop through time formulas
			nabla_p = self.d_loss(softmax(p_states[t-1]), self.data[t])

			nabla_d += nabla_p

			nabla_V += np.dot(nabla_p, h_states[t].T) 

			nabla_h = np.dot(np.dot(np.dot(self.weight_U_o.T, nabla_output), temp)\
					 +np.dot(np.dot(nabla_C, d_tanh), diag_o), nabla_h)\
					 +np.dot(self.weight_V.T, nabla_p)

			diag_f = np.diag(f_states[t-1][:,0])
			diag_i = np.diag(i_states[t-1][:,0])
			diag_g = np.diag(g_states[t-1][:,0])
			diag_o = np.diag(o_states[t-1][:,0])

			nabla_forget = np.dot(diag_f, (np.identity(self.hidden_size)-diag_f))
			nabla_input = np.dot(diag_i, (np.identity(self.hidden_size)-diag_i)) 
			nabla_gate = np.identity(self.hidden_size)-diag_g**2
			nabla_output = np.dot(diag_o, (np.identity(self.hidden_size)-diag_o))

			temp = np.diag(np.tanh(C_states[t])[:,0])
			d_tanh = np.identity(self.hidden_size) - temp**2
			diag_C = np.diag(C_states[t-1][:,0])

			nabla_temp_f = np.dot(np.dot(np.dot(np.dot(nabla_forget, d_tanh), diag_C),\
								  diag_o), nabla_h)
			nabla_temp_i = np.dot(np.dot(np.dot(np.dot(nabla_input, d_tanh), diag_g), \
								  diag_o), nabla_h)
			nabla_temp_g = np.dot(np.dot(np.dot(nabla_output, temp), diag_o), \
								  nabla_h)
			nabla_temp_o = np.dot(np.dot(np.dot(nabla_output, d_tanh), diag_i), \
								nabla_h)

			nabla_bias_f += nabla_temp_f
			nabla_bias_i += nabla_temp_i
			nabla_bias_g += nabla_temp_g
			nabla_bias_o += nabla_temp_o

			nabla_U_f += np.dot(nabla_temp_f, h_states[t-1].T) 
			nabla_U_i += np.dot(nabla_temp_i, h_states[t-1].T)  
			nabla_U_g += np.dot(nabla_temp_g, h_states[t-1].T) 
			nabla_U_o += np.dot(nabla_temp_o, h_states[t-1].T) 

			nabla_W_f += np.dot(nabla_temp_f, self.data[t-1].T)
			nabla_W_i += np.dot(nabla_temp_i, self.data[t-1].T)
			nabla_W_g += np.dot(nabla_temp_g, self.data[t-1].T)
			nabla_W_o += np.dot(nabla_temp_o, self.data[t-1].T)

			nabla_C = np.dot(np.dot(self.weight_U_f.T, nabla_forget), diag_C)\
					 +np.dot(np.dot(self.weight_U_i.T, nabla_input), diag_g)\
					 +np.dot(np.dot(self.weight_U_g.T, nabla_gate), diag_i)

		return (nabla_bias_f, nabla_bias_i, nabla_bias_g, nabla_bias_o,\
				nabla_U_f, nabla_U_i, nabla_U_g, nabla_U_o, \
				nabla_W_f, nabla_W_i, nabla_W_g, nabla_W_o, nabla_d, nabla_V)

	def d_loss(self, output_activation, y):
		"""Returns the vector of partial derivative \nabla_{o^{(t)}}L
		for the output activations of the cross entropy loss."""
		return (output_activation-y)

if __name__ == '__main__':
	location = 'C:\\Users\\Lukasz Obara\\OneDrive\\Documents\\'\
				+'Machine Learning\\Text Files\\test.csv'
	temp = np.genfromtxt(location, delimiter=',')
	my_data = [np.array(arr) for arr in temp[:, :, np.newaxis]]

	n = 20
	sequence = [my_data[i:i+n] for i in range(0, len(my_data), n)]
	lstm = LSTM(sequence[6], 100)
	lstm.train(20, 5, .15, 0.99)