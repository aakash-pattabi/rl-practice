## Learning Reinforcement Learning 

In this project, I'll be implementing a series of classic reinforcement learning (RL) algorithms, simply as a personal project to improve my own competencies in RL. I'll mostly follow the reading list from [this Reddit post](https://old.reddit.com/r/reinforcementlearning/comments/8k356e/new_phd_student_what_papers_should_i_read_first/), implementing classical control/Markov Decision Process (MDP) solver algorithms such as value iteration as well as more modern RL algorithms. 

Additionally, I'll try to discuss each paper that I read, summarize it as best as I can, and dig in, in some sense, to the intuition underlying the math. 

In order: 

* [ ] [Temporal difference learning (Sutton 1998)](https://link.springer.com/content/pdf/10.1007/BF00115009.pdf)

  1. _Temporal difference_ learning differs from supervised learning in that updates can be made to the parameters of the learned model before the final outcome is known. That is, when predicting _z_ from _t, t+1, t+2,..._, the learner can update between _t_ and _t+1_ without knowing the true label _z_. Thus, the update rule **cannot** be applied incrementally.

* [ ] [Q-learning (Watkins and Dayan 1992)](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)

* [ ] [REINFORCE algorithm (Williams 1992)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.129.8871&rep=rep1&type=pdf)

* [ ] [Prioritized sweeping (Moore and Atkeson 1993)](https://link.springer.com/content/pdf/10.1007/BF00993104.pdf)

* [ ] [Value function approximation (Boyan and Moore 1995)](http://papers.nips.cc/paper/1018-generalization-in-reinforcement-learning-safely-approximating-the-value-function.pdf)

* [ ] [TD-Gammon (Tesauro 1995)](http://enzodesiage.com/wp-content/uploads/2017/08/tesauro-tdgammon-1995.pdf)

* [ ] [POMDPs (Kaelbling et. al. 1998)](http://www.ai.mit.edu/courses/6.825/pdf/pomdp.pdf)

* [ ] [Reward shaping (Ng, Harada, and Russell 1999)](http://robotics.stanford.edu/%7Eang/papers/shaping-icml99.pdf)