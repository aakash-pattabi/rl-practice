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

class SimplePredictor(nn.Module):
	def __init__(self, input_dim, output_dim, n_hidden_neurons, activation):
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

	def forward(self, x):
		x = self.input(x)
		x = self.hidden(x)

		if self.activation is not None:
			x = self.activation(x)

		return x

class TemporalDifferenceLearner(object):
	def __init__(self, exp, lr, discount, epsilon,
		input_dim, output_dim, n_hidden_neurons, activation):

		self.exp = exp
		self.learning_rate = lr
		self.discount_rate = discount
		self.epsilon = epsilon
		self.predictor = SimplePredictor(input_dim, output_dim, n_hidden_neurons, activation)

	def epsilon_greedy_action(self, env):
		if np.random.rand() < self.epsilon:
			return (env.action_space.sample())
		n_actions = env.action_space.n
		state = env.env.s
		values = []
		for action in range(n_actions):
			s_a = torch.from_numpy(np.append(state, action)).double()
			q = self.predictor.forward(s_a)
			values.append(q)
		return np.argmax(values)

	def softmax_action():
		pass 

class SARSALearner(TemporalDifferenceLearner):
	def __init__(self):
		super().__init__()

	def update_parameters(self, s, a, r, sp, ap):

		## See pg. 10 in Geist and Pietquin (2010)
		self.predictor.zero_grad()
		s_a = torch.from_numpy(np.append(s, a)).double()
		Qs_a = self.predictor.forward(s_a)
		Qs_a.backward()

		sp_ap = torch.from_numpy(np.append(sp, ap)).double()
		Qsp_ap = self.predictor.forward(sp_ap)

		td_error = r + self.gamma * (Qsp_ap) - Qs_a

		for param in self.predictor.parameters():
			param += self.learning_rate * param.grad * td_error

############################################################################################

def main():
	env = gym.make("FrozenLake8x8-v0")
	env.reset()
	cur_obs = env.env.s

	for __ in range(10):
		# env.render()
		action = env.action_space.sample()
		next_obs, reward, done, info = env.step(action)
		print (cur_obs, action, reward, next_obs)
		cur_obs = next_obs

if __name__ == "__main__":
	main()