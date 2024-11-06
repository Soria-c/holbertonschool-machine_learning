# Deep Q\-learning

## Resources


**Read or watch**:


* [Deep Q\-Learning \- Combining Neural Networks and Reinforcement Learning](https://www.youtube.com/watch?v=wrBUkpiRvCA&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=12 "Deep Q-Learning - Combining Neural Networks and Reinforcement Learning")
* [Replay Memory Explained \- Experience for Deep Q\-Network Training](https://www.youtube.com/watch?v=Bcuj2fTH4_4&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=13 "Replay Memory Explained - Experience for Deep Q-Network Training")
* [Training a Deep Q\-Network \- Reinforcement Learning](https://www.youtube.com/watch?v=0bt0SjbS3xc&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=14 "Training a Deep Q-Network - Reinforcement Learning")
* [Training a Deep Q\-Network with Fixed Q\-targets \- Reinforcement Learning](https://www.youtube.com/watch?v=xVkPh9E9GfE&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=1  5 "Training a Deep Q-Network with Fixed Q-targets - Reinforcement Learning")
* [Deep Reinforcement Learning Libraries](https://blog.dataiku.com/on-choosing-a-deep-reinforcement-learning-library "Deep Reinforcement Learning Libraries")


**References**:


* [Gymnasium Wrappers for existing environment modification](https://gymnasium.farama.org/main/api/wrappers/ "Gymnasium Wrappers for existing environment modification")
* [Setting up anaconda for keras\-rl](https://amineneifer.medium.com/setting-up-anaconda-for-keras-rl-cec5a5c639e1 "Setting up anaconda for keras-rl")
* [How to import keras\_rl](https://stackoverflow.com/questions/76650327/how-to-fix-cannot-import-name-version-from-tensorflow-keras "How to import keras_rl")
* [keras\-rl](https://github.com/keras-rl/keras-rl "keras-rl")
    + [rl.policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py "rl.policy")
    + [rl.memory](https://github.com/keras-rl/keras-rl/blob/master/rl/memory.py "rl.memory")
    + [rl.agents.dqn](https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py "rl.agents.dqn")
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602 "Playing Atari with Deep Reinforcement Learning")


## Learning Objectives


* What is Deep Q\-learning?
* What is the policy network?
* What is replay memory?
* What is the target network?
* Why must we utilize two separate networks during training?
* What is keras\-rl? How do you use it?


## Requirements


### General


* Allowed editors: `vi`, `vim`, `emacs`
* All your files will be interpreted/compiled on Ubuntu 20\.04 LTS using `python3` (version 3\.9\)
* Your files will be executed with `numpy` (version 1\.25\.2\), `gymnasium` (version 0\.29\.1\), `keras` (version 2\.15\.0\), and `keras-rl2` (version 1\.0\.4\)
* All your files should end with a new line
* The first line of all your files should be exactly `#!/usr/bin/env python3`
* A `README.md` file, at the root of the folder of the project, is mandatory
* Your code should use the `pycodestyle` style (version 2\.11\.1\)
* All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
* All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
* All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
* All your files must be executable
* **Your code should use the minimum number of operations**


## Installing Keras\-RL



```
pip install --user keras-rl2==1.0.4
  
```

### Dependencies



```
pip install --user gymnasium[atari]==0.29.1
  pip install --user tensorflow==2.15.0
  pip install --user keras==2.15.0
  pip install --user numpy==1.25.2
  pip install --user Pillow==10.3.0
  pip install --user h5py==3.11.0
  pip install autorom[accept-rom-license]
  
```






## Tasks







### 0\. Breakout



Write a python script `train.py` that utilizes `keras`, `keras-rl2`, and `gymnasium` to train an agent that can play Atari’s Breakout:


* Your script should utilize `keras-rl2`‘s `DQNAgent`, `SequentialMemory`, and `EpsGreedyQPolicy`
* Your script should save the final policy network as `policy.h5`


Write a python script `play.py` that can display a game played by the agent trained by `train.py`:


* Your script should load the policy network saved in `policy.h5`
* Your agent should use the `GreedyQPolicy`


**HINT**: To make Gymnasium compatible with `keras-rl`, you need to update the function `reset`, `step` and `render` using the Wrappers provided by gymnasium.

