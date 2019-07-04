import gym
import statistics
import sys
from tensorboardX import SummaryWriter
from rl_agents import *

class GymEnvSimulator(object):
	def __init__(self, env, agent, run_params, log_dir):
		self.env = env
		self.agent = agent

		self.train = run_params["train"]
		self.render = run_params["render"]
		self.verbose = run_params["verbose"]

		self.n_episodes = 0
		self.writer = SummaryWriter(log_dir)

	def set_verbosity(self, verbosity):
		self.verbose = verbosity

	def set_render(self, render):
		self.render = render

	def set_train(self, train):
		self.train = train
		if self.train:
			self.agent.set_train()
		else:
			self.agent.set_eval()

	def simulate_episode(self):
		ret, steps = 0, 0
		s = self.env.reset()
		done = False
		while not done:
			a = self.agent.get_next_action(s)
			sp, r, done, info = self.env.step(a)
			ret += r
			steps += 1

			if self.train:
				self.agent.update(s, a, r, sp)

			if self.render:
				self.env.render()
				self.env.refresh(render = False)

			s = sp
		return ret, steps

	def simulate_episode_sequence(self, n_episodes):
		all_rets, all_steps = [], []
		for i in range(n_episodes):
			ret, steps = self.simulate_episode()
			all_rets.append(ret)
			all_steps.append(steps)

			# Log results to console after sequence of episodes (useful?)
			if self.verbose:
				pass
		return all_rets, all_steps

	def simulate(self, n_train_epochs, n_eval, n_train_per_epoch):
		assert n_train_epochs > 0

		train_rets, train_steps = [], []
		test_rets, test_steps = [], []

		for i in range(n_train_epochs):
			print("Training for epoch [{}/{}]".format(i, n_train_epochs), end = "\r")
			sys.stdout.flush()

			# Train for [n_train_per_epoch] episodes every epoch
			ret, steps = self.simulate_episode_sequence(n_train_per_epoch)
			train_rets += ret
			train_steps += steps

			# Evaluate for [n_eval] episodes every epoch
			self.set_train(False)
			ret, steps = self.simulate_episode_sequence(n_eval)
			test_rets += ret
			test_steps += steps
			self.set_train(True)

			# Anneal epsilon after each epoch
			self.agent.anneal_epsilon()

			# Log to Tensorboard
			self.writer.add_scalar("train_return", statistics.mean(train_rets[-n_train_per_epoch:]), i)
			self.writer.add_scalar("train_steps", statistics.mean(train_steps[-n_train_per_epoch:]), i)
			self.writer.add_scalar("test_return", statistics.mean(test_rets[-n_eval:]), i)
			self.writer.add_scalar("test_steps", statistics.mean(test_steps[-n_eval:]), i)
				
		return (train_rets, train_steps, test_rets, test_steps)

	def save_logs(self, args):
		pass

if __name__ == "__main__":
	env = gym.make("FrozenLake-v0")

	agent_params = {
		"env" : env, 
		"epsilon" : 0.9, 
		"epsilon_decay" : 0,
		"lr" : 1e-5, 
		"gamma" : 0.95
	}
	agent = TabularQAgent(**agent_params)

	run_params = {
		"train" : True, 
		"render" : False, 
		"verbose" : False
	}
	log_dir = "logs/FrozenLake-v0/4"
	simulator = GymEnvSimulator(env, agent, run_params, log_dir)

	n_train_epochs = 100000
	n_train_per_epoch = 1
	n_eval = 100

	__, __, test_rets, test_steps = simulator.simulate(n_train_epochs, n_eval, n_train_per_epoch)
	simulator.writer.close()
