import gym 
import torch

class RLAgent(object):
	def __init__(self, env):
		self.env = env

	def set_train(self):
		pass

	def set_eval(self):
		pass

	def get_next_action(self, s):
		pass

	def _get_random_action(self):
		return self.env.action_space.sample()

	def update(self, s, a, r, sp):
		pass

	def anneal_epsilon(self):
		pass

class RandomAgent(RLAgent):
	def __init__(self, env):
		super().__init__(env)

	def get_next_action(self, s):
		return self._get_random_action()

class TabularQAgent(RLAgent):
	def __init__(self, env, epsilon, epsilon_decay, lr, gamma):
		super().__init__(env)
		self.epsilon = epsilon
		self.eps_cache = epsilon
		self.epsilon_decay = epsilon_decay

		self.lr = lr
		self.gamma = gamma

		n_states = self.env.observation_space.n
		n_actions = self.env.action_space.n
		self.q_table = torch.zeros(n_states, n_actions)

	def set_train(self):
		self.epsilon = self.eps_cache

	def set_eval(self):
		self.epsilon = 0

	def get_next_action(self, s):
		if torch.rand(1) < self.epsilon:
			return self._get_random_action()
		else:
			q, action = torch.max(self.q_table[s,:], 0)
			return action.item()

	def update(self, s, a, r, sp):
		td_target = r + self.gamma*torch.max(self.q_table[sp:])
		td_error = td_target - self.q_table[s, a]
		self.q_table[s, a] += self.lr * td_error

	def anneal_epsilon(self):
		self.epsilon *= (1 - self.epsilon_decay)
		self.eps_cache = self.epsilon

class DQNAgent(RLAgent):
	def __init__(self, env, predictor, epsilon, epsilon_decay, lr, gamma):
		super().__init__(env)
		self.epsilon = epsilon
		self.eps_cache = epsilon
		self.epsilon_decay = epsilon_decay

		self.lr = lr
		self.gamma = gamma



