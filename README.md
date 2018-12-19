## Learning Reinforcement Learning 

In this project, I'll be implementing a series of classic reinforcement learning (RL) algorithms, simply as a personal project to improve my own competencies in RL. I'll mostly follow the reading list from [this Reddit post](https://old.reddit.com/r/reinforcementlearning/comments/8k356e/new_phd_student_what_papers_should_i_read_first/), implementing classical control/Markov Decision Process (MDP) solver algorithms such as value iteration as well as more modern RL algorithms. 

Additionally, I'll try to discuss each paper that I read, summarize it as best as I can, and dig in, in some sense, to the intuition underlying the math. 

In order: 

* [ ] [Temporal difference learning (Sutton 1998)](https://link.springer.com/content/pdf/10.1007/BF00115009.pdf)

  1. _Temporal difference_ learning differs from supervised learning in that updates can be made to the parameters of the learned model before the final outcome is known. That is, when predicting _z_ from _t, t+1, t+2,..._, the learner can update between _t_ and _t+1_ without knowing the true label _z_. Thus, the update rule **can** be applied incrementally, as opposed to the supervised learning setting in which it **cannot**.

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;w&space;\gets&space;w&space;&plus;&space;\alpha\sum_{t=1}^m&space;(z&space;-&space;\hat{z}_t)\nabla_w&space;\hat{z_t}&space;\indent&space;\indent&space;\&space;(1)&space;\\&space;w&space;\gets&space;w&space;&plus;&space;\alpha\sum_{t&space;=&space;1}^m&space;(\hat{z}_{t&plus;1}&space;-&space;\hat{z}_t)&space;\sum_{i&space;=&space;1}^t&space;\nabla_w&space;\hat{z}_t&space;\indent&space;(2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;w&space;\gets&space;w&space;&plus;&space;\alpha\sum_{t=1}^m&space;(z&space;-&space;\hat{z}_t)\nabla_w&space;\hat{z_t}&space;\indent&space;\indent&space;\&space;(1)&space;\\&space;w&space;\gets&space;w&space;&plus;&space;\alpha\sum_{t&space;=&space;1}^m&space;(\hat{z}_{t&plus;1}&space;-&space;\hat{z}_t)&space;\sum_{i&space;=&space;1}^t&space;\nabla_w&space;\hat{z}_t&space;\indent&space;(2)" title="\large w \gets w + \alpha\sum_{t=1}^m (z - \hat{z}_t)\nabla_w \hat{z_t} \indent \indent \ (1) \\ w \gets w + \alpha\sum_{t = 1}^m (\hat{z}_{t+1} - \hat{z}_t) \sum_{i = 1}^t \nabla_w \hat{z}_i \indent (2)" /></a>


  2. The temporal difference approach (2) (also known as a _TD(1)_ rule) saves on memory as the learner needn't remember the calculated gradients until the final outcome _z_ is known. But, it is more computationally intensive (as in the double sum). 

  3. Sutton writes, "The hallmark of temporal-difference methods is their sensitivity to changes in successive predictions rather than to overall error between predictions and the final outcome." Let's unpack that: 

  	 1. If two successive predictions at _t, t+1_ differ greatly, then the learner's weights _w_ will change greatly as well on the update. Intuitively, this is because the large temporal difference suggests that perhaps at _t_, the learner should have picked up on something that it didn't notice until _t+1_. 

  	 2. So, could we reframe temporal difference learning as a form of "smooth learning," with the hypothesis that an ideal learner only changes their beliefs **gradually** instead of **suddenly** (as the latter hints at having missed information in the environment)?

  4. We can introduce _TD(k)_ (or really _lambda_) learning where the update at _t_ privileges the most recent gradients with an exponentially decaying parameter _k_ (or _lambda_) between 0 and 1. 

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;w&space;\gets&space;w&space;&plus;&space;\alpha&space;\sum_{t&space;=&space;1}^m&space;(\hat{z}_{t&plus;1}&space;-&space;\hat{z}_t)&space;\sum_{i=1}^t&space;\lambda^{t-i}\nabla_w&space;\hat{z}_i&space;\indent&space;0&space;\leq&space;\lambda&space;\leq&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;w&space;\gets&space;w&space;&plus;&space;\alpha&space;\sum_{t&space;=&space;1}^m&space;(\hat{z}_{t&plus;1}&space;-&space;\hat{z}_t)&space;\sum_{i=1}^t&space;\lambda^{t-i}\nabla_w&space;\hat{z}_i&space;\indent&space;0&space;\leq&space;\lambda&space;\leq&space;1" title="\large w \gets w + \alpha \sum_{t = 1}^m (\hat{z}_{t+1} - \hat{z}_t) \sum_{i=1}^t \lambda^{t-i}\nabla_w \hat{z}_i \indent 0 \leq \lambda \leq 1" /></a>


  5. TD methods have an advantage over supervised learning when the data generating process is "dynamical" -- e.g. as in speech recognition, economic forecasting, where sequences of observations are highly correlated. TD can especially outshine conventional supervised learning when you have _limited new experience_ and want to avoid associating these new observations with **only** the episode outcomes that succeeded them (as opposed to with some of the encoded knowledge about the values of the adjacent states that you have _already_ calculated). 

  	1. A disadvantage of _TD(0)_ methods is that they are slow at propagating knowledge about the outcome back towards the predictions that would have been made about states observed early on. _TD(k)_ methods with _k > 0_ ensure that previous states' predictions are updates as the weights _w_ are updated. 

  6. "The expected values of the predictions found by linear TD(0) converge to the ideal predictions for data generated by absorbing (terminating, or episodic) Markov processes." Sutton denotes a TD process as linear where the hypothesis function is linear in the state vector. 

* [ ] [Q-learning (Watkins and Dayan 1992)](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)

* [ ] [REINFORCE algorithm (Williams 1992)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.129.8871&rep=rep1&type=pdf)

* [ ] [Prioritized sweeping (Moore and Atkeson 1993)](https://link.springer.com/content/pdf/10.1007/BF00993104.pdf)

* [ ] [Value function approximation (Boyan and Moore 1995)](http://papers.nips.cc/paper/1018-generalization-in-reinforcement-learning-safely-approximating-the-value-function.pdf)

* [ ] [TD-Gammon (Tesauro 1995)](http://enzodesiage.com/wp-content/uploads/2017/08/tesauro-tdgammon-1995.pdf)

* [ ] [POMDPs (Kaelbling et. al. 1998)](http://www.ai.mit.edu/courses/6.825/pdf/pomdp.pdf)

* [ ] [Reward shaping (Ng, Harada, and Russell 1999)](http://robotics.stanford.edu/%7Eang/papers/shaping-icml99.pdf)