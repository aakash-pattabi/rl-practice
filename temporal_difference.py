import gym 
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import matplotlib.pyplot as plt

############################################################################################
## In theory, this simple tabular network with one hidden layer and frozen weights in the
## input layer should yield the exact same policy (on average) as a tabular SARSA or 
## Q-learning method... 
############################################################################################

class TabularPredictor(nn.Module):
	def __init__(self, n_actions, n_states, output_dim):
		super().__init__()

		self.input = nn.Linear(n_actions + n_states, n_actions * n_states, bias = False)

		actions = np.eye(n_actions)
		states = np.eye(n_states)
		input_params = np.vstack((np.tile(actions, len(states)), np.tile(states, len(actions))))
		input_params = torch.from_numpy(input_params.T)
		self.input.weight = torch.nn.Parameter(input_params, requires_grad = False)

		self.hidden = nn.Linear(n_actions * n_states, output_dim, bias = False)
		self.double()

	def forward(self, x):
		x = self.input(x)
		x = self.hidden(x)
		return(x)

	def encode_one_hot(self, on, length):
		one_hot = np.zeros(length)
		one_hot[on] = 1
		one_hot = torch.from_numpy(one_hot)
		return one_hot

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
		x = F.relu(x)
		x = self.hidden(x)
		if self.activation is not None:
			x = self.activation(x)
		return x

'''
Argument 'predictor' is a Torch neural network, so the 'forward' function is 
defined. 

#TODO: Put an assert in or something...
'''
class TemporalDifferenceLearner(object):
	def __init__(self, exp, lr, discount, epsilon, epsilon_decay, predictor):

		self.exp = exp
		self.learning_rate = lr
		self.discount_rate = discount
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.predictor = predictor

	def greedy_action(self, env):
		n_actions = env.action_space.n
		state = env.env.s
		values = []
		for action in range(n_actions):
			s_a = torch.from_numpy(np.append(state, action)).double()
			q = self.predictor.forward(s_a)
			values.append(q)
		return np.argmax(values)

	def epsilon_greedy_action(self, env):
		if np.random.rand() < self.epsilon:
			return (env.action_space.sample())
		return self.greedy_action(env)

	def anneal_epsilon(self):
		self.epsilon *= self.epsilon_decay

	def softmax_action(self, env):
		n_actions = env.action_space.n
		state = env.env.s
		values = []
		for action in range(n_actions):
			s_a = torch.from_numpy(np.append(state, action)).double()
			q = self.predictor.forward(s_a).detach().numpy()
			values.append(q)
		values = np.array(values)
		values -= np.max(values)
		probs = np.exp(values)/np.sum(np.exp(values))
		cum_probs = np.cumsum(probs)
		return np.min(np.where(cum_probs > np.random.rand()))

class SARSALearner(TemporalDifferenceLearner):
	def __init__(self, exp, lr, discount, epsilon, epsilon_decay, predictor):

		TemporalDifferenceLearner.__init__(self, exp, lr, discount, epsilon, epsilon_decay, 
			predictor)

	def update_parameters(self, s, a, r, sp, ap):
		## See pg. 10 in Geist and Pietquin (2010)
		self.predictor.zero_grad()
		s_a = torch.from_numpy(np.append(s, a)).double()
		Qs_a = self.predictor.forward(s_a)
		Qs_a.backward()

		sp_ap = torch.from_numpy(np.append(sp, ap)).double()
		Qsp_ap = self.predictor.forward(sp_ap)

		td_error = r + self.discount_rate * (Qsp_ap) - Qs_a

		for param in self.predictor.parameters():
			if param.requires_grad:
				param = param + self.learning_rate * param.grad * td_error

############################################################################################

def main():
	REWARD = 0
	EPISODES = 0
	STEPS_IN_EPISODE = 0
	AVG_REWARD_PER_STEP = []
	EPISODE_LENGTH = []

	env = gym.make("FrozenLake8x8-v0")
	env.reset()
	s = env.env.s

	learner = SARSALearner(exp = 0, lr = 0.01, discount = 0.95, epsilon = 1, epsilon_decay = 0.99, 
		input_dim = 2, output_dim = 1, n_hidden_neurons = 100, activation = "relu")

	a = learner.epsilon_greedy_action(env)

	try:
		while EPISODES < 1000:
			sp, r, done, info = env.step(a)
			ap = learner.epsilon_greedy_action(env)
			learner.update_parameters(s, a, r, sp, ap)

			## Anneal epsilon at the end of every episode
			if done:
				env.reset()
				learner.anneal_epsilon()
				s = env.env.s
				a = learner.epsilon_greedy_action(env)
				EPISODES += 1
				AVG_REWARD_PER_STEP.append(1.0 * REWARD/STEPS_IN_EPISODE)
				EPISODE_LENGTH.append(STEPS_IN_EPISODE)
				print ("Completed episode %d with reward %d in %d steps" 
					% (EPISODES, REWARD, STEPS_IN_EPISODE))
				STEPS_IN_EPISODE = 0
				REWARD = 0

			s, a = sp, ap
			STEPS_IN_EPISODE += 1

	except KeyboardInterrupt:
		pass

	plt.plot(range(EPISODES), AVG_REWARD_PER_STEP)
	plt.title("SARSA: Avg. reward per step")
	plt.xlabel("Episode")
	plt.ylabel("Avg. reward in episode")
	plt.savefig("SARSA_AvgReward_v0.png")

	plt.clf()
	plt.plot(range(EPISODES), EPISODE_LENGTH)
	plt.title("SARSA: Avg. episode length")
	plt.xlabel("Episode")
	plt.ylabel("Episode length")
	plt.savefig("SARSA_EpisodeLength_v0.png")

	'''
	## Play game to visualize results of learner
	env.reset()
	while True: 
		a = learner.greedy_action(env)
		env.render()
		sp, r, done, info = env.step(a)
		if done: 
			break
	'''

if __name__ == "__main__":
	# main()

	net = TabularPredictor(3, 2, 1)

	for param in net.parameters():
		print(param)

	x = torch.zeros(5).double()
	x[1], x[4] = 1, 1
	print(x)

	out = net.forward(x)
	print(out)
