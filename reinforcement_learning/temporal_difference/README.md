
# Temporal Difference

## Resources


**Read or watch**:


* [RL Course by David Silver \- Lecture 4: Model\-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA "RL Course by David Silver - Lecture 4: Model-Free Prediction")
* [RL Course by David Silver \- Lecture 5: Model Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=6 "RL Course by David Silver - Lecture 5: Model Free Control")
* [Temporal Difference Learning (including Q\-Learning)](https://www.youtube.com/watch?v=AJiG3ykOxmY "Temporal Difference Learning (including Q-Learning) ")
* [Temporal\-Difference Learning](https://www.tu-chemnitz.de/informatik/KI/scripts/ws0910/ml09_6.pdf "Temporal-Difference Learning")
* [Intro to reinforcement learning: temporal difference learning, SARSA vs. Q\-learning](https://towardsdatascience.com/intro-to-reinforcement-learning-temporal-difference-learning-sarsa-vs-q-learning-8b4184bb4978 "Intro to reinforcement learning: temporal difference learning, SARSA vs. Q-learning")
* [Temporal difference reinforcement learning](https://gibberblot.github.io/rl-notes/single-agent/temporal-difference-learning.html "Temporal difference reinforcement learning")
* [Reinforcement Learning: Temporal Difference (TD) Learning](https://www.lancaster.ac.uk/stor-i-student-sites/jordan-j-hood/2021/04/12/reinforcement-learning-temporal-difference-td-learning/ "Reinforcement Learning: Temporal Difference (TD) Learning")
* [On\-Policy TD Control](https://paperswithcode.com/methods/category/on-policy-td-control "On-Policy TD Control")


**Definitions to skim:**


* [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method "Monte Carlo method")
* [Temporal difference learning](https://en.wikipedia.org/wiki/Temporal_difference_learning "Temporal difference learning")
* [State–action–reward–state–action](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action "State–action–reward–state–action")


## Learning Objectives


* What is Monte Carlo?
* What is Temporal Difference?
* What is bootstrapping?
* What is n\-step temporal difference?
* What is TD(λ)?
* What is an eligibility trace?
* What is SARSA? SARSA(λ)? SARSAMAX?
* What is ‘on\-policy’ vs ‘off\-policy’?


## Requirements


### General


* Allowed editors: `vi`, `vim`, `emacs`
* All your files will be interpreted/compiled on Ubuntu 20\.04 LTS using `python3` (version 3\.9\)
* Your files will be executed with `numpy` (version 1\.25\.2\), and `gymnasium` (version 0\.29\.1\)
* All your files should end with a new line
* The first line of all your files should be exactly `#!/usr/bin/env python3`
* A `README.md` file, at the root of the folder of the project, is mandatory
* Your code should use the `pycodestyle` style (version 2\.11\.1\)
* All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
* All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
* All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
* All your files must be executable
* **Your code should use the minimum number of operations**




#### Question \#0



Which of the following are true of Q\-learning?



* It is an on\-policy method
* ***It is a type of TD algorithm***
* It is the same as Sarsa(0\)
* ***It is more efficient than Monte Carlo***







#### Question \#1



Monte Carlo is the equivalent of: 



* TD(0\)
* TD(λ)
* ***TD(1\)***
* None of the above







#### Question \#2



Eligibility traces are used in:



* Off\-policy algorithms
* ***Backward facing algorithms***
* Exact\-online algorithms
* None of the above











## Tasks







### 0\. Monte Carlo



Write the function `def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):` that performs the Monte Carlo algorithm:


* `env` is environment instance
* `V` is a `numpy.ndarray` of shape `(s,)` containing the value estimate
* `policy` is a function that takes in a state and returns the next action to take
* `episodes` is the total number of episodes to train over
* `max_steps` is the maximum number of steps per episode
* `alpha` is the learning rate
* `gamma` is the discount rate
* Returns: `V`, the updated value estimate



```
$ cat 0-main.py
  #!/usr/bin/env python3
  
  import gymnasium as gym
  import numpy as np
  import random
  monte_carlo = __import__('0-monte_carlo').monte_carlo
  
  
  def set_seed(env, seed=0):
      env.reset(seed=seed)
      np.random.seed(seed)
      random.seed(seed)
  
  env = gym.make('FrozenLake8x8-v1')
  set_seed(env, 0)
  
  LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
  
  def policy(s):
      p = np.random.uniform()
      if p > 0.5:
          if s % 8 != 7 and env.unwrapped.desc[s // 8, s % 8 + 1] != b'H':
              return RIGHT
          elif s // 8 != 7 and env.unwrapped.desc[s // 8 + 1, s % 8] != b'H':
              return DOWN
          elif s // 8 != 0 and env.unwrapped.desc[s // 8 - 1, s % 8] != b'H':
              return UP
          else:
              return LEFT
      else:
          if s // 8 != 7 and env.unwrapped.desc[s // 8 + 1, s % 8] != b'H':
              return DOWN
          elif s % 8 != 7 and env.unwrapped.desc[s // 8, s % 8 + 1] != b'H':
              return RIGHT
          elif s % 8 != 0 and env.unwrapped.desc[s // 8, s % 8 - 1] != b'H':
              return LEFT
          else:
              return UP
  
  V = np.where(env.unwrapped.desc == b'H', -1, 1).reshape(64).astype('float64')
  np.set_printoptions(precision=4)
  
  print(monte_carlo(env, V, policy).reshape((8, 8)))
  
  $ ./0-main.py
  [[ 0.4305  0.729   0.6561  0.729   0.729   0.9     0.5905  0.5314]
   [ 0.2542  0.5314  0.5905  0.81    0.6561  0.3874  0.4783  0.3874]
   [ 0.729   0.2824  0.3487 -1.      1.      0.4783  0.4305  0.4305]
   [ 1.      0.4305  0.2288  0.5905  0.9    -1.      0.4783  0.4783]
   [ 1.      0.6561  0.5905 -1.      1.      1.      0.729   0.729 ]
   [ 1.     -1.     -1.      1.      1.      1.     -1.      0.9   ]
   [ 1.     -1.      1.      1.     -1.      1.     -1.      1.    ]
   [ 1.      1.      1.     -1.      1.      1.      1.      1.    ]]
  $
  
```



### 1\. TD(λ)



Write the function `def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):` that performs the TD(λ) algorithm:


* `env` is the environment instance
* `V` is a `numpy.ndarray` of shape `(s,)` containing the value estimate
* `policy` is a function that takes in a state and returns the next action to take
* `lambtha` is the eligibility trace factor
* `episodes` is the total number of episodes to train over
* `max_steps` is the maximum number of steps per episode
* `alpha` is the learning rate
* `gamma` is the discount rate
* Returns: `V`, the updated value estimate



```
$ cat 1-main.py
  #!/usr/bin/env python3
  
  import gymnasium as gym
  import numpy as np
  import random
  
  td_lambtha = __import__('1-td_lambtha').td_lambtha
  
  def set_seed(env, seed=0):
      env.reset(seed=seed)
      np.random.seed(seed)
      random.seed(seed)
  
  env = gym.make('FrozenLake8x8-v1')
  set_seed(env, 0)
  
  LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
  
  def policy(s):
      p = np.random.uniform()
      if p > 0.5:
          if s % 8 != 7 and env.unwrapped.desc[s // 8, s % 8 + 1] != b'H':
              return RIGHT
          elif s // 8 != 7 and env.unwrapped.desc[s // 8 + 1, s % 8] != b'H':
              return DOWN
          elif s // 8 != 0 and env.unwrapped.desc[s // 8 - 1, s % 8] != b'H':
              return UP
          else:
              return LEFT
      else:
          if s // 8 != 7 and env.unwrapped.desc[s // 8 + 1, s % 8] != b'H':
              return DOWN
          elif s % 8 != 7 and env.unwrapped.desc[s // 8, s % 8 + 1] != b'H':
              return RIGHT
          elif s % 8 != 0 and env.unwrapped.desc[s // 8, s % 8 - 1] != b'H':
              return LEFT
          else:
              return UP
  
  V = np.where(env.unwrapped.desc == b'H', -1, 1).reshape(64).astype('float64')
  np.set_printoptions(precision=4)
  
  print(td_lambtha(env, V, policy, 0.9).reshape((8, 8)))
  
  $ ./1-main.py
  [[-0.8197 -0.8389 -0.8462 -0.8123 -0.764  -0.7915 -0.6869 -0.7692]
   [-0.8647 -0.8895 -0.9099 -0.8847 -0.7526 -0.7783 -0.7098 -0.7406]
   [-0.8833 -0.9191 -0.9517 -1.     -0.7988 -0.7229 -0.4986 -0.6516]
   [-0.8959 -0.9218 -0.9463 -0.9682 -0.9561 -1.     -0.4612 -0.6806]
   [-0.9252 -0.9356 -0.9476 -1.     -0.9247 -0.862  -0.5377 -0.4785]
   [-0.9255 -1.     -1.      0.8029 -0.8679 -0.8339 -1.     -0.222 ]
   [-0.9466 -1.     -0.5755  0.3756 -1.     -0.2279 -1.     -0.1423]
   [-0.9336 -0.9307 -0.8733 -1.      1.      0.6173  0.7587  1.    ]]
  $
  
```



### 2\. SARSA(λ)








Write the function `def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):` that performs SARSA(λ):


* `env` is the environment instance
* `Q` is a `numpy.ndarray` of shape `(s,a)` containing the Q table
* `lambtha` is the eligibility trace factor
* `episodes` is the total number of episodes to train over
* `max_steps` is the maximum number of steps per episode
* `alpha` is the learning rate
* `gamma` is the discount rate
* `epsilon` is the initial threshold for epsilon greedy
* `min_epsilon` is the minimum value that epsilon should decay to
* `epsilon_decay` is the decay rate for updating epsilon between episodes
* Returns: `Q`, the updated Q table



```
$ cat 2-main.py
  #!/usr/bin/env python3
  
  import gymnasium as gym
  import numpy as np
  import random
  
  sarsa_lambtha = __import__('2-sarsa_lambtha').sarsa_lambtha
  
  def set_seed(env, seed=0):
      env.reset(seed=seed)
      np.random.seed(seed)
      random.seed(seed)
  
  env = gym.make('FrozenLake8x8-v1')
  set_seed(env, 0)
  Q = np.random.uniform(size=(64, 4))
  np.set_printoptions(precision=4)
  
  print(sarsa_lambtha(env, Q, 0.9))
  
  $ ./2-main.py
  [[0.5939 0.605  0.6805 0.5798]
   [0.5721 0.5776 0.6192 0.6942]
   [0.6087 0.6004 0.7354 0.6133]
   [0.6706 0.6354 0.7223 0.6587]
   [0.6482 0.7571 0.6666 0.6166]
   [0.7161 0.6898 0.8691 0.714 ]
   [0.6959 0.8638 0.6768 0.6857]
   [0.7118 0.8854 0.7088 0.7061]
   [0.6071 0.6716 0.605  0.5905]
   [0.6074 0.6083 0.5915 0.6844]
   [0.5969 0.6212 0.6988 0.5741]
   [0.4929 0.5095 0.5925 0.7254]
   [0.6432 0.6636 0.7762 0.6647]
   [0.7079 0.7114 0.7491 0.7092]
   [0.7419 0.8355 0.7271 0.7145]
   [0.7099 0.6346 0.9006 0.6994]
   [0.6662 0.6403 0.6191 0.6272]
   [0.6244 0.6049 0.6467 0.6784]
   [0.7141 0.6006 0.56   0.5567]
   [0.2828 0.1202 0.2961 0.1187]
   [0.4374 0.4289 0.7727 0.4781]
   [0.6865 0.723  0.8322 0.6671]
   [0.7605 0.9185 0.78   0.774 ]
   [0.7596 0.926  0.8646 0.6766]
   [0.6776 0.6374 0.7619 0.612 ]
   [0.8105 0.665  0.6429 0.6962]
   [0.6182 0.6804 0.746  0.6744]
   [0.5483 0.8008 0.6152 0.6648]
   [0.5916 0.5622 0.7997 0.6698]
   [0.8811 0.5813 0.8817 0.6925]
   [0.7989 0.7784 0.7826 0.8925]
   [0.7805 1.0058 0.7466 0.8156]
   [0.7246 0.8337 0.6811 0.7201]
   [0.8719 0.727  0.7114 0.7395]
   [0.7465 0.714  0.7256 0.6768]
   [0.8965 0.3676 0.4359 0.8919]
   [0.7657 0.7057 0.4065 0.831 ]
   [0.7314 0.7647 0.3496 0.8833]
   [0.5315 0.7346 0.4517 0.9636]
   [0.8489 1.1193 0.8106 0.6702]
   [0.6877 0.7543 0.6909 0.758 ]
   [0.9755 0.8558 0.0117 0.36  ]
   [0.73   0.1716 0.521  0.0543]
   [0.2    0.0185 0.8511 0.2854]
   [0.3454 0.918  0.7127 0.2646]
   [0.3377 0.8517 0.5772 0.391 ]
   [0.9342 0.614  0.5356 0.5899]
   [1.359  0.6305 0.5683 0.4644]
   [0.252  0.4873 0.6923 0.5272]
   [0.2274 0.2544 0.058  0.4344]
   [0.3118 0.6379 0.3778 0.1796]
   [0.0247 0.0672 0.7562 0.4537]
   [0.5366 0.8967 0.9903 0.2169]
   [0.6631 0.3272 0.0207 0.7852]
   [0.32   0.3835 0.5883 0.831 ]
   [0.7281 1.6395 0.3659 0.8018]
   [0.2949 0.4697 0.6345 0.2786]
   [0.5212 0.4958 0.2947 0.2342]
   [0.4649 0.0257 0.2271 0.4268]
   [0.3742 0.4636 0.2776 0.5868]
   [0.8639 0.1175 0.5174 0.1321]
   [0.7169 0.3961 0.5654 0.1833]
   [0.1448 0.4881 0.3556 0.9404]
   [0.7653 0.7487 0.9037 0.0834]]
  $
  
```