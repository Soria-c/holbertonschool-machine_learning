# Q\-learning


## Resources


**Read or watch**:


* [MIT 6\.S191: Reinforcement Learning](https://www.youtube.com/watch?v=8JVRbHAVCws "MIT 6.S191: Reinforcement Learning")
* [An introduction to Reinforcement Learning](https://www.youtube.com/watch?v=JgvyzIkgxF0 "An introduction to Reinforcement Learning")
* [An Introduction to Q\-Learning: A Tutorial For Beginners](https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial "An Introduction to Q-Learning: A Tutorial For Beginners")
* [Q\-Learning](https://www.geeksforgeeks.org/q-learning-in-python/ "Q-Learning")
* [Markov Decision Processes (MDPs) \- Structuring a Reinforcement Learning Problem](https://www.youtube.com/watch?v=my207WNoeyA&t=18s "Markov Decision Processes (MDPs) - Structuring a Reinforcement Learning Problem")
* [Expected Return \- What Drives a Reinforcement Learning Agent in an MDP](https://www.youtube.com/watch?v=a-SnJtmBtyA&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&t=23s "Expected Return - What Drives a Reinforcement Learning Agent in an MDP")
* [Policies and Value Functions \- Good Actions for a Reinforcement Learning Agent](https://www.youtube.com/watch?v=eMxOGwbdqKY&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&t=23s "Policies and Value Functions - Good Actions for a Reinforcement Learning Agent")
* [What do Reinforcement Learning Algorithms Learn \- Optimal Policies](https://www.youtube.com/watch?v=rP4oEpQbDm4&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&t=20s "What do Reinforcement Learning Algorithms Learn - Optimal Policies")
* [Q\-Learning Explained \- A Reinforcement Learning Technique](https://www.youtube.com/watch?v=qhRNvCVVJaA&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&t=26s "Q-Learning Explained - A Reinforcement Learning Technique")
* [Exploration vs. Exploitation \- Learning the Optimal Reinforcement Learning Policy](https://www.youtube.com/watch?v=mo96Nqlo1L8&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&t=24s "Exploration vs. Exploitation - Learning the Optimal Reinforcement Learning Policy")
* [Q\-Learning Tutorial 1: Train Gymnasium FrozenLake\-v1 with Python Reinforcement Learning](https://www.youtube.com/watch?v=ZhoIgo3qqLU "Q-Learning Tutorial 1: Train Gymnasium FrozenLake-v1 with Python Reinforcement Learning")
* [Q\-Learning: Implementation](https://wandb.ai/cosmo3769/Q-Learning/reports/Q-Learning-Implementation---Vmlldzo1OTUxNTI4 "Q-Learning: Implementation")
* [Markov Decision Processes](https://www.youtube.com/watch?v=KovN7WKI9Y0 "Markov Decision Processes")


**Definitions to skim:**


* [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning "Reinforcement Learning")
* [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process "Markov Decision Process")
* [Q\-learning](https://en.wikipedia.org/wiki/Q-learning "Q-learning")


**References**:


* [Gymnasium](https://gymnasium.farama.org/ "Gymnasium")
* [Gymnasium: Frozen Lake env](https://gymnasium.farama.org/environments/toy_text/frozen_lake/ "Gymnasium: Frozen Lake env")
* [Gymnasium: Frozenlake benchmark](https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/ "Gymnasium: Frozenlake benchmark")
* [Gymnasium: Env](https://gymnasium.farama.org/api/env/#gymnasium.Env "Gymnasium: Env")


## Learning Objectives


* What is a Markov Decision Process?
* What is an environment?
* What is an agent?
* What is a state?
* What is a policy function?
* What is a value function? a state\-value function? an action\-value function?
* What is a discount factor?
* What is the Bellman equation?
* What is epsilon greedy?
* What is Q\-learning?


## More Info


### Installing Gymnasium 0\.29\.1



```
pip install --user gymnasium==0.29.1
  
```

### Dependencies (that should already be installed)



```
pip install --user Pillow==10.3.0
pip install --user h5py==3.11.0
  
```

#### Question \#0



What is reinforcement learning?



* A type of supervised learning, because the rewards supervise the learning
* A type of unsupervised learning, because there are no labels for each action
* ***Its own subcategory of machine learning***







#### Question \#1



What is an environment?



* ***The place in which actions can be performed***
* A description of what the agent sees
* A list of actions that can be performed
* A description of which actions the agent should perform







#### Question \#2



An agent chooses its action based on:



* ***The current state***
* ***The value function***
* ***The policy function***
* ***The previous reward***







#### Question \#3



What is a policy function?



* A description of how the agent should be rewarded
* ***A description of how the agent should behave***
* A description of how the agent could be rewarded in the future
* ***A function that is learned***
* A function that is set at the beginning







#### Question \#4



What is a value function?



* A description of how the agent should be rewarded
* A description of how the agent should behave
* ***A description of how the agent could be rewarded in the ***future
* ***A function that is learned***
* A function that is set at the beginning







#### Question \#5



What is epsilon\-greedy?



* A type of policy function
* A type of value function
* A way to balance policy and value functions
* ***A balance exploration and exploitation***







#### Question \#6



What is Q\-learning?



* ***A reinforcement learning algorithm***
* A deep reinforcement learning algorithm
* ***A value\-based learning algorithm***
* A policy\-based learning algorithm
* A model\-based approach











## Tasks







### 0\. Load the Environment



Write a function `def load_frozen_lake(desc=None, map_name=None, is_slippery=False):` that loads the pre\-made `FrozenLakeEnv` evnironment from `gymnasium`:


* `desc` is either `None` or a list of lists containing a custom description of the map to load for the environment
* `map_name` is either `None` or a string containing the pre\-made map to load
* *Note: If both `desc` and `map_name` are `None`, the environment will load a randomly generated 8x8 map*
* `is_slippery` is a boolean to determine if the ice is slippery
* Returns: the environment



```
$ cat 0-main.py
  #!/usr/bin/env python3
  
  load_frozen_lake = __import__('0-load_env').load_frozen_lake
  
  env = load_frozen_lake()
  print(env.unwrapped.desc) 
  print(len(env.unwrapped.P[0][0]))
  print(env.unwrapped.P[0][0])
  
  env = load_frozen_lake(is_slippery=True)
  print(env.unwrapped.desc)
  print(len(env.unwrapped.P[0][0]))
  print(env.unwrapped.P[0][0])
  
  desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
  env = load_frozen_lake(desc=desc)
  print(env.unwrapped.desc)
  
  env = load_frozen_lake(map_name='4x4')
  print(env.unwrapped.desc)
  $ ./0-main.py
  [[b'S' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
   [b'H' b'F' b'H' b'F' b'F' b'H' b'F' b'F']
   [b'F' b'F' b'H' b'H' b'F' b'F' b'H' b'F']
   [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'H']
   [b'F' b'H' b'F' b'F' b'F' b'F' b'F' b'F']
   [b'H' b'F' b'F' b'F' b'F' b'F' b'F' b'H']
   [b'F' b'H' b'F' b'F' b'F' b'H' b'F' b'F']
   [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'G']]
  1
  [(1.0, 0, 0.0, False)]
  [[b'S' b'F' b'F' b'F' b'F' b'F' b'H' b'H']
   [b'F' b'F' b'H' b'F' b'H' b'F' b'F' b'H']
   [b'H' b'F' b'H' b'F' b'F' b'F' b'F' b'F']
   [b'F' b'H' b'F' b'F' b'F' b'F' b'F' b'F']
   [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'H']
   [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'F']
   [b'F' b'F' b'F' b'H' b'H' b'F' b'F' b'F']
   [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'G']]
  3
  [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 8, 0.0, False)]
  [[b'S' b'F' b'F']
   [b'F' b'H' b'H']
   [b'F' b'F' b'G']]
  [[b'S' b'F' b'F' b'F']
   [b'F' b'H' b'F' b'H']
   [b'F' b'F' b'F' b'H']
   [b'H' b'F' b'F' b'G']]
  $
  
```

### 1\. Initialize Q\-table


Write a function `def q_init(env):` that initializes the Q\-table:


* `env` is the `FrozenLakeEnv` instance
* Returns: the Q\-table as a `numpy.ndarray` of zeros



```
$ cat 1-main.py
  #!/usr/bin/env python3
  
  load_frozen_lake = __import__('0-load_env').load_frozen_lake
  q_init = __import__('1-q_init').q_init
  
  env = load_frozen_lake()
  Q = q_init(env)
  print(Q.shape)
  env = load_frozen_lake(is_slippery=True)
  Q = q_init(env)
  print(Q.shape)
  desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
  env = load_frozen_lake(desc=desc)
  Q = q_init(env)
  print(Q.shape)
  env = load_frozen_lake(map_name='4x4')
  Q = q_init(env)
  print(Q.shape)
  $ ./1-main.py
  (64, 4)
  (64, 4)
  (9, 4)
  (16, 4)
  $
  
```

### 2\. Epsilon Greedy

Write a function `def epsilon_greedy(Q, state, epsilon):` that uses epsilon\-greedy to determine the next action:


* `Q` is a `numpy.ndarray` containing the q\-table
* `state` is the current state
* `epsilon` is the epsilon to use for the calculation
* You should sample `p` with `numpy.random.uniformn` to determine if your algorithm should explore or exploit
* If exploring, you should pick the next action with `numpy.random.randint` from all possible actions
* Returns: the next action index



```
$ cat 2-main.py
  #!/usr/bin/env python3
  
  load_frozen_lake = __import__('0-load_env').load_frozen_lake
  q_init = __import__('1-q_init').q_init
  epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
  import numpy as np
  
  desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
  env = load_frozen_lake(desc=desc)
  Q = q_init(env)
  Q[7] = np.array([0.5, 0.7, 1, -1])
  np.random.seed(0)
  print(epsilon_greedy(Q, 7, 0.5))
  np.random.seed(1)
  print(epsilon_greedy(Q, 7, 0.5))
  $ ./2-main.py
  2
  0
  $
  
```

### 3\. Q\-learning

Write the function `def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):` that performs Q\-learning:


* `env` is the `FrozenLakeEnv` instance
* `Q` is a `numpy.ndarray` containing the Q\-table
* `episodes` is the total number of episodes to train over
* `max_steps` is the maximum number of steps per episode
* `alpha` is the learning rate
* `gamma` is the discount rate
* `epsilon` is the initial threshold for epsilon greedy
* `min_epsilon` is the minimum value that `epsilon` should decay to
* `epsilon_decay` is the decay rate for updating `epsilon` between episodes
* When the agent falls in a hole, the reward should be updated to be `-1`
* You should use `epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy`
* Returns: `Q, total_rewards`
    + `Q` is the updated Q\-table
    + `total_rewards` is a list containing the rewards per episode



```
$ cat 3-main.py
  #!/usr/bin/env python3
  
  load_frozen_lake = __import__('0-load_env').load_frozen_lake
  q_init = __import__('1-q_init').q_init
  train = __import__('3-q_learning').train
  import numpy as np
  
  np.random.seed(0)
  desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
  env = load_frozen_lake(desc=desc)
  Q = q_init(env)
  
  Q, total_rewards  = train(env, Q)
  print(Q)
  split_rewards = np.split(np.array(total_rewards), 10)
  for i, rewards in enumerate(split_rewards):
      print((i+1) * 500, ':', np.mean(rewards))
  $ ./3-main.py
  [[0.96059595 0.970299   0.95098641 0.96059508]
   [0.96059578 0.         0.00914612 0.43792863]
   [0.17824547 0.         0.         0.        ]
   [0.97029808 0.9801     0.         0.96059035]
   [0.         0.         0.         0.        ]
   [0.         0.         0.         0.        ]
   [0.9800999  0.98009991 0.99       0.97029895]
   [0.98009936 0.98999805 1.         0.        ]
   [0.         0.         0.         0.        ]]
  500 : 0.918
  1000 : 0.962
  1500 : 0.948
  2000 : 0.946
  2500 : 0.948
  3000 : 0.964
  3500 : 0.95
  4000 : 0.934
  4500 : 0.928
  5000 : 0.934
  $
  
```

*Note : The output may vary based on your implemntation but for every run, you should have the same result*




### 4\. Play


Write a function `def play(env, Q, max_steps=100):` that has the trained agent play an episode:


* `env` is the `FrozenLakeEnv` instance
* `Q` is a `numpy.ndarray` containing the Q\-table
* `max_steps` is the maximum number of steps in the episode
    + You need to update `0-load_env.py` to add `render_mode="ansi"`
    + Each state of the board should be displayed via the console
    + You should always exploit the Q\-table
    + Ensure that the final state of the environment is also displayed after the episode concludes.
* Returns: The total rewards for the episode and a list of rendered outputs representing the board state at each step.



```
$ cat 4-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train
play = __import__('4-play').play

import numpy as np

np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

Q, _ = train(env, Q)

env.reset()
total_rewards, rendered_outputs = play(env, Q)

print(f'Total Rewards: {total_rewards}')
for output in rendered_outputs:
    print(output)
$ ./4-main.py
Total Rewards: 1.0

`S`FF
FHH
FFG
  (Down)
SFF
`F`HH
FFG
  (Down)
SFF
FHH
`F`FG
  (Right)
SFF
FHH
F`F`G
  (Right)
SFF
FHH
FF`G`

$

```
