# Multivariate Probability

## Resources

* [Joint Probability Distributions](/rltoken/4PVdDmgggOYmzUy2eBLyUg "Joint Probability Distributions")
* [Multivariate Gaussian distributions](https://www.youtube.com/watch?v=eho8xH3E6mE "Multivariate Gaussian distributions")
* [The Multivariate Gaussian Distribution](https://cs229.stanford.edu/section/gaussians.pdf "The Multivariate Gaussian Distribution")
* [An Introduction to Variance, Covariance \& Correlation](https://www.alchemer.com/resources/blog/variance-covariance-correlation/ "An Introduction to Variance, Covariance & Correlation")
* [Variance\-covariance matrix using matrix notation of factor analysis](https://www.youtube.com/watch?v=G16c2ZODcg8 "Variance-covariance matrix using matrix notation of factor analysis")


* [Carl Friedrich Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss)
* [Joint probability distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution)
* [Covariance](https://en.wikipedia.org/wiki/Covariance)
* [Covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix)


* [numpy.cov](/rltoken/N8SO85DloHLfmi9yQVThnQ "numpy.cov")
* [numpy.corrcoef](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html "numpy.corrcoef")
* [numpy.linalg.det](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html "numpy.linalg.det")
* [numpy.linalg.inv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html "numpy.linalg.inv")
* [numpy.random.multivariate\_normal](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html "numpy.random.multivariate_normal")
## Learning Objectives



## Tasks

### 0\. Mean and Covariance

Write a function `def mean_cov(X):` that calculates the mean and covariance of a data set:

* `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set:
	+ `n` is the number of data points
	+ `d` is the number of dimensions in each data point
	+ If `X` is not a 2D `numpy.ndarray`, raise a `TypeError` with the message `X must be a 2D numpy.ndarray`
	+ If `n` is less than 2, raise a `ValueError` with the message `X must contain multiple data points`
* Returns: `mean`, `cov`:
	+ `mean` is a `numpy.ndarray` of shape `(1, d)` containing the mean of the data set
	+ `cov` is a `numpy.ndarray` of shape `(d, d)` containing the covariance matrix of the data set
* You are not allowed to use the function `numpy.cov`


```python
alexa@ubuntu-xenial:multivariate_prob$ cat 0-main.py 
 #!/usr/bin/env python3
 
 if __name__ == '__main__':
 import numpy as np
 mean_cov = __import__('0-mean_cov').mean_cov
 
 np.random.seed(0)
 X = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000)
 mean, cov = mean_cov(X)
 print(mean)
 print(cov)
 alexa@ubuntu-xenial:multivariate_prob$ ./0-main.py 
 [[12.04341828 29.92870885 10.00515808]]
 [[ 36.2007391 -29.79405239 15.37992641]
 [-29.79405239 97.77730626 -20.67970134]
 [ 15.37992641 -20.67970134 24.93956823]]
 alexa@ubuntu-xenial:multivariate_prob$
 ``` 
### 1\. Correlation

Write a function `def correlation(C):` that calculates a correlation matrix:

* `C` is a `numpy.ndarray` of shape `(d, d)` containing a covariance matrix
	+ `d` is the number of dimensions
	+ If `C` is not a `numpy.ndarray`, raise a `TypeError` with the message `C must be a numpy.ndarray`
	+ If `C` does not have shape `(d, d)`, raise a `ValueError` with the message `C must be a 2D square matrix`
* Returns a `numpy.ndarray` of shape `(d, d)` containing the correlation matrix


```python
alexa@ubuntu-xenial:multivariate_prob$ cat 1-main.py 
 #!/usr/bin/env python3
 
 if __name__ == '__main__':
 import numpy as np
 correlation = __import__('1-correlation').correlation
 
 C = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
 Co = correlation(C)
 print(C)
 print(Co)
 alexa@ubuntu-xenial:multivariate_prob$ ./1-main.py 
 [[ 36 -30 15]
 [-30 100 -20]
 [ 15 -20 25]]
 [[ 1. -0.5 0.5]
 [-0.5 1. -0.4]
 [ 0.5 -0.4 1. ]]
 alexa@ubuntu-xenial:multivariate_prob$
 ``` 
### 2\. Initialize

Create the class `MultiNormal` that represents a Multivariate Normal distribution:

* class constructor `def __init__(self, data):`
	+ `data` is a `numpy.ndarray` of shape `(d, n)` containing the data set:
	+ `n` is the number of data points
	+ `d` is the number of dimensions in each data point
	+ If `data` is not a 2D `numpy.ndarray`, raise a `TypeError` with the message `data must be a 2D numpy.ndarray`
	+ If `n` is less than 2, raise a `ValueError` with the message `data must contain multiple data points`
* Set the public instance variables:
	+ `mean` \- a `numpy.ndarray` of shape `(d, 1)` containing the mean of `data`
	+ `cov` \- a `numpy.ndarray` of shape `(d, d)` containing the covariance matrix `data`
* You are not allowed to use the function `numpy.cov`


```python
alexa@ubuntu-xenial:multivariate_prob$ cat 2-main.py 
 #!/usr/bin/env python3
 
 if __name__ == '__main__':
 import numpy as np
 from multinormal import MultiNormal
 
 np.random.seed(0)
 data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
 mn = MultiNormal(data)
 print(mn.mean)
 print(mn.cov)
 alexa@ubuntu-xenial:multivariate_prob$ ./2-main.py 
 [[12.04341828]
 [29.92870885]
 [10.00515808]]
 [[ 36.2007391 -29.79405239 15.37992641]
 [-29.79405239 97.77730626 -20.67970134]
 [ 15.37992641 -20.67970134 24.93956823]]
 alexa@ubuntu-xenial:multivariate_prob$
 ``` 
### 3\. PDF

Update the class `MultiNormal`:

* public instance method `def pdf(self, x):` that calculates the PDF at a data point:
	+ `x` is a `numpy.ndarray` of shape `(d, 1)` containing the data point whose PDF should be calculated
		- `d` is the number of dimensions of the `Multinomial` instance
	+ If `x` is not a `numpy.ndarray`, raise a `TypeError` with the message `x must be a numpy.ndarray`
	+ If `x` is not of shape `(d, 1)`, raise a `ValueError` with the message `x must have the shape ({d}, 1)`
	+ Returns the value of the PDF
	+ You are not allowed to use the function `numpy.cov`


```python
alexa@ubuntu-xenial:multivariate_prob$ cat 3-main.py 
 #!/usr/bin/env python3
 
 if __name__ == '__main__':
 import numpy as np
 from multinormal import MultiNormal
 
 np.random.seed(0)
 data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
 mn = MultiNormal(data)
 x = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 1).T
 print(x)
 print(mn.pdf(x))
 alexa@ubuntu-xenial:multivariate_prob$ ./3-main.py 
 [[ 8.20311936]
 [32.84231319]
 [ 9.67254478]]
 0.00022930236202143827
 alexa@ubuntu-xenial:multivariate_prob$
 ``` 
