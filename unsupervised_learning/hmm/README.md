# Hidden Markov Chains
## Resources
* [Markov property](https://en.wikipedia.org/wiki/Markov_property)
* [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)
## Learning Objectives
* What is the Markov property?
* What is a Markov chain?
* What is a state?
* What is a transition probability/matrix?
* What is a stationary state?
* What is a regular Markov chain?
* How to determine if a transition matrix is regular
* What is an absorbing state?
* What is a transient state?
* What is a recurrent state?
What is an absorbing Markov chain?
* What is a Hidden Markov Model?
* What is a hidden state?
* What is an observation?
* What is an emission probability/matrix?
* What is a Trellis diagram?
* What is the Forward algorithm and how do you implement it?
* What is decoding?
* What is the Viterbi algorithm and how do you implement it?
* What is the Forward-Backward algorithm and how do you implement it?
* What is the Baum-Welch algorithm and how do you implement it?
## Tasks
### 0. Markov Chain
Write the function **def markov_chain(P, s, t=1):** that determines the probability of a markov chain being in a particular state after a specified number of iterations:
* **P** is a square 2D **numpy.ndarray** of shape **(n, n) **representing the transition matrix
    * **P[i, j]** is the probability of transitioning from state **i** to state **j**
    * **n** is the number of states in the markov chain
* **s** is a **numpy.ndarray** of shape **(1, n)** representing the probability of starting in each state
* **t** is the number of iterations that the markov chain has been through
* Returns: a **numpy.ndarray** of shape **(1, n)** representing the probability of being in a specific state after **t** iterations, or **None** on failure
```python
#!/usr/bin/env python3

import numpy as np
markov_chain = __import__('0-markov_chain').markov_chain

if __name__ == "__main__":
    P = np.array([[0.25, 0.2, 0.25, 0.3], [0.2, 0.3, 0.2, 0.3], [0.25, 0.25, 0.4, 0.1], [0.3, 0.3, 0.1, 0.3]])
    s = np.array([[1, 0, 0, 0]])
    print(markov_chain(P, s, 300))
```
```python
[[0.2494929  0.26335362 0.23394185 0.25321163]]
```