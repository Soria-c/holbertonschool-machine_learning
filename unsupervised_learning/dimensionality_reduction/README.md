# Dimensionality Reduction

## Resources

* [Dimensionality Reduction For Dummies — Part 1: Intuition](https://towardsdatascience.com/https-medium-com-abdullatif-h-dimensionality-reduction-for-dummies-part-1-a8c9ec7b7e79 "Dimensionality Reduction For Dummies — Part 1: Intuition")
* [Singular Value Decomposition](https://www.youtube.com/watch?v=P5mlg91as1c "Singular Value Decomposition")
* [Understanding SVD (Singular Value Decomposition)](https://towardsdatascience.com/svd-8c2f72e264f "Understanding SVD (Singular Value Decomposition)")
* [Intuitively, what is the difference between Eigendecomposition and Singular Value Decomposition?](https://math.stackexchange.com/questions/320220/intuitively-what-is-the-difference-between-eigendecomposition-and-singular-valu "Intuitively, what is the difference between Eigendecomposition and Singular Value Decomposition?")
* [Dimensionality Reduction: Principal Components Analysis, Part 1](https://www.youtube.com/watch?v=ZqXnPcyIAL8 "Dimensionality Reduction: Principal Components Analysis, Part 1")
* [Dimensionality Reduction: Principal Components Analysis, Part 2](https://www.youtube.com/watch?v=NUn6WeFM5cM "Dimensionality Reduction: Principal Components Analysis, Part 2")
* [StatQuest: t\-SNE, Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM "StatQuest: t-SNE, Clearly Explained")
* [t\-SNE tutorial Part1](https://www.youtube.com/watch?v=ohQXphVSEQM "t-SNE tutorial Part1")
* [t\-SNE tutorial Part2](https://www.youtube.com/watch?v=W-9L6v_rFIE "t-SNE tutorial Part2")
* [How to Use t\-SNE Effectively](https://distill.pub/2016/misread-tsne/ "How to Use t-SNE Effectively")


* [Dimensionality Reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)
* [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
* [Eigendecomposition of a matrix](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix)
* [Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
* [Manifold](https://en.wikipedia.org/wiki/Manifold)
* [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#:~:text=In%20the%20simple%20case%2C%20a,mechanics%2C%20neuroscience%20and%20machine%20learning.)
* [T-distributed stochastic neighbor embedding](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)



* [numpy.cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html "numpy.cumsum")
* [Visualizing Data using t\-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf "Visualizing Data using t-SNE") (paper)
* [Visualizing Data Using t\-SNE](https://www.youtube.com/watch?v=RJVL80Gg3lA "Visualizing Data Using t-SNE") (video)


* [Kernel principal component analysis](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis	)
* [Nonlinear Dimensionality Reduction: KPCA](https://www.youtube.com/watch?v=HbDHohXPLnU)

## Learning Objectives

* What is eigendecomposition?
* What is singular value decomposition?
* What is the difference between eig and svd?
* What is dimensionality reduction and what are its purposes?
* What is principal components analysis (PCA)?
* What is t-distributed stochastic neighbor embedding (t-SNE)?
* What is a manifold?
* What is the difference between linear and non-linear dimensionality reduction?
* Which techniques are linear/non-linear?

## Data

Please test your main files with the following data:

* mnist2500_X.txt
* mnist2500_labels.txt

## Performance between SVD and EIG

Here a graph of execution time (Y-axis) for the number of iteration (X-axis) - red line is EIG and blue line is SVG

![](./data/df2eac7a51b56139b4a179a83398b18fbda8868c.png)


## Tasks

### 0\. PCA

Write a function `def pca(X, var=0.95):` that performs PCA on a dataset:

* `X` is a `numpy.ndarray` of shape `(n, d)` where:
	+ `n` is the number of data points
	+ `d` is the number of dimensions in each point
	+ all dimensions have a mean of 0 across all data points
* `var` is the fraction of the variance that the PCA transformation should maintain
* Returns: the weights matrix, `W`, that maintains `var` fraction of `X`‘s original variance
* `W` is a `numpy.ndarray` of shape `(d, nd)` where `nd` is the new dimensionality of the transformed `X`


```python
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 0-main.py 
 #!/usr/bin/env python3
 
 import numpy as np
 pca = __import__('0-pca').pca
 
 np.random.seed(0)
 a = np.random.normal(size=50)
 b = np.random.normal(size=50)
 c = np.random.normal(size=50)
 d = 2 * a
 e = -5 * b
 f = 10 * c
 
 X = np.array([a, b, c, d, e, f]).T
 m = X.shape[0]
 X_m = X - np.mean(X, axis=0)
 W = pca(X_m)
 T = np.matmul(X_m, W)
 print(T)
 X_t = np.matmul(T, W.T)
 print(np.sum(np.square(X_m - X_t)) / m)
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./0-main.py 
 [[-16.71379391 3.25277063 -3.21956297]
 [ 16.22654311 -0.7283969 -0.88325252]
 [ 15.05945199 3.81948929 -1.97153621]
 [ -7.69814111 5.49561088 -4.34581561]
 [ 14.25075197 1.37060228 -4.04817187]
 [-16.66888233 -3.77067823 2.6264981 ]
 [ 6.71765183 0.18115089 -1.91719288]
 [ 10.20004065 -0.84380128 0.44754302]
 [-16.93427229 1.72241573 0.9006236 ]
 [-12.4100987 0.75431367 -0.36518129]
 [-16.40464248 1.98431953 0.34907508]
 [ -6.69439671 1.30624703 -2.77438892]
 [ 10.84363895 4.99826372 -1.36502623]
 [-17.2656016 7.29822621 0.63226953]
 [ 5.32413372 -0.54822516 -0.79075935]
 [ -5.63240657 1.50278876 -0.27590797]
 [ -7.63440366 7.72788006 -2.58344477]
 [ 4.3348786 -2.14969035 0.61262033]
 [ -3.95417052 4.22254889 -0.14601319]
 [ -6.59947069 -1.00867621 2.29551761]
 [ -0.78942283 -4.15454151 5.87117533]
 [ 13.62292856 0.40038586 -1.36043631]
 [ 0.03536684 -5.85950737 -1.86196569]
 [-11.1841298 5.20313078 2.37753549]
 [ 9.62095425 -1.17179699 -4.97535412]
 [ 3.85296648 3.55808 3.65166717]
 [ 6.57934417 4.87503426 0.30243418]
 [-16.17025935 1.49358788 1.0663259 ]
 [ -4.33639793 1.26186205 -2.99149191]
 [ -1.52947063 -0.39342225 -2.96475006]
 [ 9.80619496 6.65483286 0.07714817]
 [ -2.45893463 -4.89091813 -0.6918453 ]
 [ 9.56282904 -1.8002211 2.06720323]
 [ 1.70293073 7.68378254 5.03581954]
 [ 9.58030378 -6.97453776 0.64558546]
 [ -3.41279182 -10.07660784 -0.39277019]
 [ -2.74983634 -6.25461193 -2.65038235]
 [ 4.54987003 1.28692201 -2.40001675]
 [ -1.81149682 5.16735962 1.4245976 ]
 [ 13.97823555 -4.39187437 0.57600155]
 [ 17.39107161 3.26808567 2.50429006]
 [ -1.25835112 -6.60720376 3.24220508]
 [ 1.06405562 -1.25980089 4.06401644]
 [ -3.44578711 -5.21002054 -4.20836152]
 [-21.1181523 -3.72353504 1.6564066 ]
 [ -6.56723647 -4.31268383 1.22783639]
 [ 11.77670231 0.67338386 2.94885044]
 [ -7.89417224 -9.82300322 -1.69743681]
 [ 15.87543091 0.3804009 3.67627751]
 [ 7.38044431 -1.58972122 0.60154138]]
 1.7353180054998176e-29
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$
 ``` 
### 1\. PCA v2

Write a function `def pca(X, ndim):` that performs PCA on a dataset:

* `X` is a `numpy.ndarray` of shape `(n, d)` where:
	+ `n` is the number of data points
	+ `d` is the number of dimensions in each point
* `ndim` is the new dimensionality of the transformed `X`
* Returns: `T`, a `numpy.ndarray` of shape `(n, ndim)` containing the transformed version of `X`


```python
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 1-main.py 
 #!/usr/bin/env python3
 
 import numpy as np
 pca = __import__('1-pca').pca
 
 X = np.loadtxt("mnist2500_X.txt")
 print('X:', X.shape)
 print(X)
 T = pca(X, 50)
 print('T:', T.shape)
 print(T)
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./1-main.py 
 X: (2500, 784)
 [[1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]
 ...
 [1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]]
 T: (2500, 50)
 [[-0.61344587 1.37452188 -1.41781926 ... -0.42685217 0.02276617
 0.1076424 ]
 [-5.00379081 1.94540396 1.49147124 ... 0.26249077 -0.4134049
 -1.15489853]
 [-0.31463237 -2.11658407 0.36608266 ... -0.71665401 -0.18946283
 0.32878802]
 ...
 [ 3.52302175 4.1962009 -0.52129062 ... -0.24412645 0.02189273
 0.19223197]
 [-0.81387035 -2.43970416 0.33244717 ... -0.55367626 -0.64632309
 0.42547833]
 [-2.25717018 3.67177791 2.83905021 ... -0.35014766 -0.01807652
 0.31548087]]
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$
 ``` 
### 2\. Initialize t\-SNE

Write a function `def P_init(X, perplexity):` that initializes all variables required to calculate the P affinities in [t\-SNE](/rltoken/S5DY2PH7B5h_xUW0kF9Ltw "t-SNE"):

* `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset to be transformed by t\-SNE
	+ `n` is the number of data points
	+ `d` is the number of dimensions in each point
* `perplexity` is the perplexity that all Gaussian distributions should have
* Returns: `(D, P, betas, H)`
	+ `D`: a `numpy.ndarray` of shape `(n, n)` that calculates the squared pairwise distance between two data points
		- The diagonal of `D` should be `0`s
	+ `P`: a `numpy.ndarray` of shape `(n, n)` initialized to all `0`‘s that will contain the P affinities
	+ `betas`: a `numpy.ndarray` of shape `(n, 1)` initialized to all `1`’s that will contain all of the `beta` values
		- ![](https://latex.codecogs.com/gif.latex?\beta_{i}&space;=&space;\frac{1}{2{\sigma_{i}}^{2}&space;} "\beta_{i} = \frac{1}{2{\sigma_{i}}^{2} }")
	+ `H` is the Shannon entropy for `perplexity` perplexity with a base of 2


```python
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 2-main.py 
 #!/usr/bin/env python3
 
 import numpy as np
 pca = __import__('1-pca').pca
 P_init = __import__('2-P_init').P_init
 
 X = np.loadtxt("mnist2500_X.txt")
 X = pca(X, 50)
 D, P, betas, H = P_init(X, 30.0)
 print('X:', X.shape)
 print(X)
 print('D:', D.shape)
 print(D.round(2))
 print('P:', P.shape)
 print(P)
 print('betas:', betas.shape)
 print(betas)
 print('H:', H)
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./2-main.py 
 X: (2500, 50)
 [[-0.61344587 1.37452188 -1.41781926 ... -0.42685217 0.02276617
 0.1076424 ]
 [-5.00379081 1.94540396 1.49147124 ... 0.26249077 -0.4134049
 -1.15489853]
 [-0.31463237 -2.11658407 0.36608266 ... -0.71665401 -0.18946283
 0.32878802]
 ...
 [ 3.52302175 4.1962009 -0.52129062 ... -0.24412645 0.02189273
 0.19223197]
 [-0.81387035 -2.43970416 0.33244717 ... -0.55367626 -0.64632309
 0.42547833]
 [-2.25717018 3.67177791 2.83905021 ... -0.35014766 -0.01807652
 0.31548087]]
 D: (2500, 2500)
 [[ 0. 107.88 160.08 ... 129.62 127.61 121.67]
 [107.88 0. 170.97 ... 142.58 147.69 116.64]
 [160.08 170.97 0. ... 138.66 110.26 115.89]
 ...
 [129.62 142.58 138.66 ... 0. 156.12 109.99]
 [127.61 147.69 110.26 ... 156.12 0. 114.08]
 [121.67 116.64 115.89 ... 109.99 114.08 0. ]]
 P: (2500, 2500)
 [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
 betas: (2500, 1)
 [[1.]
 [1.]
 [1.]
 ...
 [1.]
 [1.]
 [1.]]
 H: 4.906890595608519
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$
 ``` 
### 3\. Entropy

Write a function `def HP(Di, beta):` that calculates the Shannon entropy and P affinities relative to a data point:

* `Di` is a `numpy.ndarray` of shape `(n - 1,)` containing the pariwise distances between a data point and all other points except itself
	+ `n` is the number of data points
* `beta` is a `numpy.ndarray` of shape `(1,)` containing the beta value for the Gaussian distribution
* Returns: `(Hi, Pi)`
	+ `Hi`: the Shannon entropy of the points
	+ `Pi`: a `numpy.ndarray` of shape `(n - 1,)` containing the P affinities of the points
* *Hint: see page 4 of [t\-SNE](/https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf "t-SNE")*


```python
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 3-main.py 
 #!/usr/bin/env python3
 
 import numpy as np
 pca = __import__('1-pca').pca
 P_init = __import__('2-P_init').P_init
 HP = __import__('3-entropy').HP
 
 X = np.loadtxt("mnist2500_X.txt")
 X = pca(X, 50)
 D, P, betas, _ = P_init(X, 30.0)
 H0, P[0, 1:] = HP(D[0, 1:], betas[0])
 print(H0)
 print(P[0])
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./3-main.py 
 0.057436093636173254
 [0.00000000e+00 3.74413188e-35 8.00385528e-58 ... 1.35664798e-44
 1.00374765e-43 3.81537517e-41]
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$
 ``` 
### 4\. P affinities

Write a function `def P_affinities(X, tol=1e-5, perplexity=30.0):` that calculates the symmetric P affinities of a data set:

* `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset to be transformed by t\-SNE
	+ `n` is the number of data points
	+ `d` is the number of dimensions in each point
* `perplexity` is the perplexity that all Gaussian distributions should have
* `tol` is the maximum tolerance allowed (inclusive) for the difference in Shannon entropy from `perplexity` for all Gaussian distributions
* You should use `P_init = __import__('2-P_init').P_init` and `HP = __import__('3-entropy').HP`
* Returns: `P`, a `numpy.ndarray` of shape `(n, n)` containing the symmetric P affinities
* *Hint 1: See page 6 of [t\-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf "t-SNE")*
* *Hint 2: For this task, you will need to perform a binary search on each point’s distribution to find the correct value of `beta` that will give a Shannon Entropy `H` within the tolerance (Think about why we analyze the Shannon entropy instead of perplexity). Since beta can be in the range `(0, inf)`, you will have to do a binary search with the `high` and `low` initially set to `None`. If in your search, you are supposed to increase/decrease `beta` to `high`/`low` but they are still set to `None`, you should double/half the value of `beta` instead.*


```python
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 4-main.py 
 #!/usr/bin/env python3
 
 import numpy as np
 pca = __import__('1-pca').pca
 P_affinities = __import__('4-P_affinities').P_affinities
 
 X = np.loadtxt("mnist2500_X.txt")
 X = pca(X, 50)
 P = P_affinities(X)
 print('P:', P.shape)
 print(P)
 print(np.sum(P))
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./4-main.py 
 P: (2500, 2500)
 [[0.00000000e+00 7.40714907e-10 9.79862968e-13 ... 2.37913671e-11
 1.22844912e-10 1.75011944e-10]
 [7.40714907e-10 0.00000000e+00 1.68735728e-13 ... 2.11150140e-12
 1.05003596e-11 2.42913116e-10]
 [9.79862968e-13 1.68735728e-13 0.00000000e+00 ... 2.41827214e-11
 3.33128330e-09 1.25696380e-09]
 ...
 [2.37913671e-11 2.11150140e-12 2.41827214e-11 ... 0.00000000e+00
 3.62850172e-12 4.11671350e-10]
 [1.22844912e-10 1.05003596e-11 3.33128330e-09 ... 3.62850172e-12
 0.00000000e+00 6.70800054e-10]
 [1.75011944e-10 2.42913116e-10 1.25696380e-09 ... 4.11671350e-10
 6.70800054e-10 0.00000000e+00]]
 1.0000000000000004
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$
 ``` 
### 5\. Q affinities

Write a function `def Q_affinities(Y):` that calculates the Q affinities:

* `Y` is a `numpy.ndarray` of shape `(n, ndim)` containing the low dimensional transformation of `X`
	+ `n` is the number of points
	+ `ndim` is the new dimensional representation of `X`
* Returns: `Q, num`
	+ `Q` is a `numpy.ndarray` of shape `(n, n)` containing the Q affinities
	+ `num` is a `numpy.ndarray` of shape `(n, n)` containing the numerator of the Q affinities
* *Hint: See page 7 of [t\-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf "t-SNE")*


```python
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 5-main.py 
 #!/usr/bin/env python3
 
 import numpy as np
 Q_affinities = __import__('5-Q_affinities').Q_affinities
 
 np.random.seed(0)
 Y = np.random.randn(2500, 2)
 Q, num = Q_affinities(Y)
 print('num:', num.shape)
 print(num)
 print(np.sum(num))
 print('Q:', Q.shape)
 print(Q)
 print(np.sum(Q))
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./5-main.py 
 num: (2500, 2500)
 [[0. 0.1997991 0.34387413 ... 0.08229525 0.43197616 0.29803545]
 [0.1997991 0. 0.08232739 ... 0.0780192 0.36043254 0.20418429]
 [0.34387413 0.08232739 0. ... 0.07484357 0.16975081 0.17792688]
 ...
 [0.08229525 0.0780192 0.07484357 ... 0. 0.13737822 0.22790422]
 [0.43197616 0.36043254 0.16975081 ... 0.13737822 0. 0.65251175]
 [0.29803545 0.20418429 0.17792688 ... 0.22790422 0.65251175 0. ]]
 2113140.980877581
 Q: (2500, 2500)
 [[0.00000000e+00 9.45507652e-08 1.62731275e-07 ... 3.89445137e-08
 2.04423728e-07 1.41039074e-07]
 [9.45507652e-08 0.00000000e+00 3.89597234e-08 ... 3.69209645e-08
 1.70567198e-07 9.66259681e-08]
 [1.62731275e-07 3.89597234e-08 0.00000000e+00 ... 3.54181605e-08
 8.03310395e-08 8.42001935e-08]
 ...
 [3.89445137e-08 3.69209645e-08 3.54181605e-08 ... 0.00000000e+00
 6.50113847e-08 1.07850932e-07]
 [2.04423728e-07 1.70567198e-07 8.03310395e-08 ... 6.50113847e-08
 0.00000000e+00 3.08787608e-07]
 [1.41039074e-07 9.66259681e-08 8.42001935e-08 ... 1.07850932e-07
 3.08787608e-07 0.00000000e+00]]
 1.0000000000000004
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$
 ``` 
### 6\. Gradients

Write a function `def grads(Y, P):` that calculates the gradients of Y:

* `Y` is a `numpy.ndarray` of shape `(n, ndim)` containing the low dimensional transformation of `X`
* `P` is a `numpy.ndarray` of shape `(n, n)` containing the P affinities of `X`
* Do not multiply the gradients by the scalar 4 as described in the paper’s equation
* Returns: `(dY, Q)`
	+ `dY` is a `numpy.ndarray` of shape `(n, ndim)` containing the gradients of `Y`
	+ `Q` is a `numpy.ndarray` of shape `(n, n)` containing the Q affinities of `Y`
* You may use `Q_affinities = __import__('5-Q_affinities').Q_affinities`
* *Hint: See page 8 of [t\-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf "t-SNE")*


```python
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 6-main.py 
 #!/usr/bin/env python3
 
 import numpy as np
 pca = __import__('1-pca').pca
 P_affinities = __import__('4-P_affinities').P_affinities
 grads = __import__('6-grads').grads
 
 np.random.seed(0)
 X = np.loadtxt("mnist2500_X.txt")
 X = pca(X, 50)
 P = P_affinities(X)
 Y = np.random.randn(X.shape[0], 2)
 dY, Q = grads(Y, P)
 print('dY:', dY.shape)
 print(dY)
 print('Q:', Q.shape)
 print(Q)
 print(np.sum(Q))
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./6-main.py 
 dY: (2500, 2)
 [[ 1.28824814e-05 1.55400363e-05]
 [ 3.21435525e-05 4.35358938e-05]
 [-1.02947106e-05 3.53998421e-07]
 ...
 [-2.27447049e-05 -3.05191863e-06]
 [ 9.69379032e-06 1.00659610e-06]
 [ 5.75113416e-05 7.65517123e-09]]
 Q: (2500, 2500)
 [[0.00000000e+00 9.45507652e-08 1.62731275e-07 ... 3.89445137e-08
 2.04423728e-07 1.41039074e-07]
 [9.45507652e-08 0.00000000e+00 3.89597234e-08 ... 3.69209645e-08
 1.70567198e-07 9.66259681e-08]
 [1.62731275e-07 3.89597234e-08 0.00000000e+00 ... 3.54181605e-08
 8.03310395e-08 8.42001935e-08]
 ...
 [3.89445137e-08 3.69209645e-08 3.54181605e-08 ... 0.00000000e+00
 6.50113847e-08 1.07850932e-07]
 [2.04423728e-07 1.70567198e-07 8.03310395e-08 ... 6.50113847e-08
 0.00000000e+00 3.08787608e-07]
 [1.41039074e-07 9.66259681e-08 8.42001935e-08 ... 1.07850932e-07
 3.08787608e-07 0.00000000e+00]]
 1.0000000000000004
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$
 ``` 
### 7\. Cost

Write a function `def cost(P, Q):` that calculates the cost of the t\-SNE transformation:

* `P` is a `numpy.ndarray` of shape `(n, n)` containing the P affinities
* `Q` is a `numpy.ndarray` of shape `(n, n)` containing the Q affinities
* Returns: `C`, the cost of the transformation
* *Hint 1: See page 5 of [t\-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf "t-SNE")*
* *Hint 2: Watch out for division by `0` errors! Take the minimum of all values in p and q with almost `0` (ex. `1e-12`)*


```python
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 7-main.py 
 #!/usr/bin/env python3
 
 import numpy as np
 pca = __import__('1-pca').pca
 P_affinities = __import__('4-P_affinities').P_affinities
 grads = __import__('6-grads').grads
 cost = __import__('7-cost').cost
 
 np.random.seed(0)
 X = np.loadtxt("mnist2500_X.txt")
 X = pca(X, 50)
 P = P_affinities(X)
 Y = np.random.randn(X.shape[0], 2)
 _, Q = grads(Y, P)
 C = cost(P, Q)
 print(C)
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./7-main.py 
 4.531113944164376
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$
 ``` 
### 8\. t\-SNE

Write a function `def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):` that performs a t\-SNE transformation:

* `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset to be transformed by t\-SNE
	+ `n` is the number of data points
	+ `d` is the number of dimensions in each point
* `ndims` is the new dimensional representation of `X`
* `idims` is the intermediate dimensional representation of `X` after PCA
* `perplexity` is the perplexity
* `iterations` is the number of iterations
* `lr` is the learning rate
* Every 100 iterations, not including 0, print `Cost at iteration {iteration}: {cost}`
	+ `{iteration}` is the number of times Y has been updated and `{cost}` is the corresponding cost
* After every iteration, `Y` should be re\-centered by subtracting its mean
* Returns: `Y`, a `numpy.ndarray` of shape `(n, ndim)` containing the optimized low dimensional transformation of `X`
* You should use:
	+ `pca = __import__('1-pca').pca`
	+ `P_affinities = __import__('4-P_affinities').P_affinities`
	+ `grads = __import__('6-grads').grads`
	+ `cost = __import__('7-cost').cost`
* For the first 100 iterations, perform early exaggeration with an exaggeration of 4
* `a(t)` \= 0\.5 for the first 20 iterations and 0\.8 thereafter
* *Hint 1: See Algorithm 1 on page 9 of [t\-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf "t-SNE"). But WATCH OUT! There is a mistake in the gradient descent step*
* *Hint 2: See Section 3\.4 starting on page 9 of [t\-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf "t-SNE") for early exaggeration*


```python
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 8-main.py 
 #!/usr/bin/env python3
 
 import matplotlib.pyplot as plt
 import numpy as np
 tsne = __import__('8-tsne').tsne
 
 np.random.seed(0)
 X = np.loadtxt("mnist2500_X.txt")
 labels = np.loadtxt("mnist2500_labels.txt")
 Y = tsne(X, perplexity=50.0, iterations=3000, lr=750)
 plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
 plt.colorbar()
 plt.title('t-SNE')
 plt.show()
 alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./8-main.py 
 Cost at iteration 100: 15.132745380504451
 Cost at iteration 200: 1.4499349051185884
 Cost at iteration 300: 1.2991961074009266
 Cost at iteration 400: 1.2255530221811528
 Cost at iteration 500: 1.179753264451479
 Cost at iteration 600: 1.1476306791330366
 Cost at iteration 700: 1.123501502573646
 Cost at iteration 800: 1.1044968276172735
 Cost at iteration 900: 1.0890468673949145
 Cost at iteration 1000: 1.0762018736143146
 Cost at iteration 1100: 1.0652921250043619
 Cost at iteration 1200: 1.0558751316523136
 Cost at iteration 1300: 1.047653338870073
 Cost at iteration 1400: 1.0403981880716473
 Cost at iteration 1500: 1.033935359326665
 Cost at iteration 1600: 1.0281287524465708
 Cost at iteration 1700: 1.0228885344794134
 Cost at iteration 1800: 1.0181265576736775
 Cost at iteration 1900: 1.0137760713813615
 Cost at iteration 2000: 1.009782545181553
 Cost at iteration 2100: 1.0061007125574222
 Cost at iteration 2200: 1.0026950513450206
 Cost at iteration 2300: 0.9995335333268901
 Cost at iteration 2400: 0.9965894332394158
 Cost at iteration 2500: 0.9938399255561283
 Cost at iteration 2600: 0.9912653473151115
 Cost at iteration 2700: 0.9888485527807156
 Cost at iteration 2800: 0.9865746432480398
 Cost at iteration 2900: 0.9844307720012038
 Cost at iteration 3000: 0.9824051809484152
 ``` 
 ![](./data/840193a578ae3df79c7d.png)

**Awesome! We can see pretty good clusters! For comparison, here’s how PCA performs on the same dataset:**

```python
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat pca.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
pca = __import__('1-pca').pca

X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
Y = pca(X, 2)
plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.colorbar()
plt.title('PCA')
plt.show()
```
![](./data/a00ee1c23a3b8c0199fa.png)