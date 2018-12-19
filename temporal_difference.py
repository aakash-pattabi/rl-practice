import gym 
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

############################################################################################
## Use an arbitrary neural network (can be quite simple, as in a multilayer perceptron)
## to take advantage of automatic gradient calculation when calculating the temporal 
## difference weight updates
##
## Also allows for playing around with different kinds of hypothesis functions, as well 
## as interaction b/w temporal difference methods and deep learning
############################################################################################

class Predictor(nn.Module):
	def __init__(self, 
				 input_dim, output_dim, n_hidden_neurons, 
				 activation):

		super().__init__()
		assert(activation in ["relu", "sigmoid", "softmax", "linear"])

		activations = {}
		activations["relu"] = F.relu 
		activations["sigmoid"] = F.sigmoid
		activations["softmax"] = F.softmax

		self.activation = None if activation == "linear" else activations[activation]
		self.input = nn.Linear(input_dim, n_hidden_neurons)
		self.hidden = nn.Linear(n_hidden_neurons, output_dim)
		self.double()

	def forward(self):
		x = self.input(x)
		x = self.hidden(x)

		if self.activation is not None:
			x = self.activation(x)

class TemporalDifferenceLearner(object):
	def __init__(self, exp = 1, lr = 0.01, 
				 input_dim, output_dim, n_hidden_neurons, activation):
		self.exp = exp
		self.learning_rate = lr
		self.predictor = Predictor(input_dim, output_dim, n_hidden_neurons, activation)

	def update(self, guess, next_guess):

############################################################################################

def main():
	env = gym.make("CartPole-v0")
	env.reset()

if __name__ == "__main__":
	main()