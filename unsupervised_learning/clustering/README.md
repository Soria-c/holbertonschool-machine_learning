


![Project badge](/assets/pathway/004_color-d2fbcfb42ba7481834896ecc89a6f0ae65762b3c1096691dd0f820f7f29e3389.png)97%# Clustering

* Master
* By: Alexa Orrico, Software Engineer at Holberton School
* Weight: 5
* Your score will be updated as you progress.
* Manual QA review was done by Davis Joseph on Aug 4, 2024 1:30 PM




* [Description](#description)





[Go to tasks](#)

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/8/95b077928a7de094f94f.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T024523Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=0fae08f138ab3cbf3ebc80e43b1e5b38505c3d5cba785fc22f8ca72863bd162e)


## Resources


**Read or watch**:


* [Understanding K\-means Clustering in Machine Learning](/rltoken/EAXlZJ0bKUCKEq4EtGY-qw "Understanding K-means Clustering in Machine Learning")
* [K\-means clustering: how it works](/rltoken/aYepPDMz50VnaPRWvLCsaw "K-means clustering: how it works")
* [How many clusters?](/rltoken/Fevvf-bo4eRoBtWGOXHyEQ "How many clusters?")
* [Bimodal distribution](/rltoken/5LTLlNrt01Zo8vrLpoxh5Q "Bimodal distribution")
* [Gaussian Mixture Model](/rltoken/wdP_gXMOYcwdrZf2tUAbcA "Gaussian Mixture Model")
* [EM algorithm: how it works](/rltoken/Pxpz0YdPgQ7fMNwmmBqubQ "EM algorithm: how it works")
* [Expectation Maximization: how it works](/rltoken/cRLxzBGxP3azH_CRJc16UQ "Expectation Maximization: how it works")
* [Mixture Models 4: multivariate Gaussians](/rltoken/NgMYsEWFvXcucKs3mlrjJA "Mixture Models 4: multivariate Gaussians")
* [Mixture Models 5: how many Gaussians?](/rltoken/UWxhOekJcpi8VXblD-3Bmw "Mixture Models 5: how many Gaussians?")
* [Gaussian Mixture Model (GMM) using Expectation Maximization (EM) Technique](/rltoken/Haunsj7G9qfZJPU0VrWsSw "Gaussian Mixture Model (GMM) using Expectation Maximization (EM) Technique")
* [What is Hierarchical Clustering?](/rltoken/gkceP2cIyayGcPUiVbI4ew "What is Hierarchical Clustering?")
* [Hierarchical Clustering](/rltoken/WGN3N5hKvoZisqKUU4TyiA "Hierarchical Clustering")


**Definitions to skim:**


* [Cluster analysis](/rltoken/68QHL_iKDqdoqdYjIW0ewg "Cluster analysis")
* [K\-means clustering](/rltoken/PtICix0s3UhjYTZw3CltqQ "K-means clustering")
* [Multimodal distribution](/rltoken/ygIxvNgsKTfFMG2RCl7OHw "Multimodal distribution")
* [Mixture\_model](/rltoken/IXjB2UuXan1nBKFDiwZDXQ "Mixture_model")
* [Expectation–maximization algorithm](/rltoken/6NBsQ-sHY0QxeuXyC9miXg "Expectation–maximization algorithm")
* [Hierarchical clustering](/rltoken/qbjmk2RY4dRtyc6xXXArFw "Hierarchical clustering")
* [Ward’s method](/rltoken/mcVrzOnzDW1EmxwoqYBGoA "Ward's method")
* [Cophenetic](/rltoken/cOSU06G9d1cfST-HdjAk9A "Cophenetic")


**References**:


* [scikit\-learn](/rltoken/NVIyKlOpSzs4GYN0q0oMUg "scikit-learn")
* [Clustering](/rltoken/I97CEUyWU9dalnWwNJJJfw "Clustering")
* [sklearn.cluster.KMeans](/rltoken/GfTFWRdgdGqglY0SgxXa1Q "sklearn.cluster.KMeans")
* [Gaussian mixture models](/rltoken/FSc0zTdnUqCnaI3z8zDRQw "Gaussian mixture models")
* [sklearn.mixture.GaussianMixture](/rltoken/PI3owlEQeqVUsNUvjPH51w "sklearn.mixture.GaussianMixture")
* [scipy](/rltoken/ozuj4Zy0hIzmQT-6IQKeQw "scipy")
* [scipy.cluster.hierarchy](/rltoken/3cYjV2amyFvwJrXNzDXV0A "scipy.cluster.hierarchy")
	+ [scipy.cluster.hierarchy.linkage](/rltoken/vB6zNONPppbIm9VnJY1GnA "scipy.cluster.hierarchy.linkage")
	+ [scipy.cluster.hierarchy.fcluster](/rltoken/SJbR2DWb6WoKjO1to-WR_A "scipy.cluster.hierarchy.fcluster")
	+ [scipy.cluster.hierarchy.dendrogram](/rltoken/Ea5pz1r-Omh4yDzp7Tuiwg "scipy.cluster.hierarchy.dendrogram")


## Learning Objectives


* What is a multimodal distribution?
* What is a cluster?
* What is cluster analysis?
* What is “soft” vs “hard” clustering?
* What is K\-means clustering?
* What are mixture models?
* What is a Gaussian Mixture Model (GMM)?
* What is the Expectation\-Maximization (EM) algorithm?
* How to implement the EM algorithm for GMMs
* What is cluster variance?
* What is the mountain/elbow method?
* What is the Bayesian Information Criterion?
* How to determine the correct number of clusters
* What is Hierarchical clustering?
* What is Agglomerative clustering?
* What is Ward’s method?
* What is Cophenetic distance?
* What is `scikit-learn`?
* What is `scipy`?


## Requirements


### General


* Allowed editors: `vi`, `vim`, `emacs`
* All your files will be interpreted/compiled on Ubuntu 20\.04 LTS using `python3` (version 3\.9\)
* Your files will be executed with `numpy` (version 1\.25\.2\), `sklearn` (version 1\.5\.0\), and `scipy` (version 1\.11\.4\)
* All your files should end with a new line
* The first line of all your files should be exactly `#!/usr/bin/env python3`
* A `README.md` file, at the root of the folder of the project, is mandatory
* Your code should use the `pycodestyle` style (version 2\.11\.1\)
* All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
* All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
* All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
* Unless otherwise noted, you are not allowed to import any module except `import numpy as np`
* All your files must be executable
* **Your code should use the minimum number of operations**


## Installing Scikit\-Learn 1\.5\.0



```
pip install --user scikit-learn==1.5.0
	
```

## Installing Scipy 1\.11\.4


`scipy` should have already been installed with `matplotlib` and `numpy`, but just in case:



```
pip install --user scipy==1.11.4
	
```






## Tasks







### 0\. Initialize K\-means




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def initialize(X, k):` that initializes cluster centroids for K\-means:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset that will be used for K\-means clustering
	+ `n` is the number of data points
	+ `d` is the number of dimensions for each data point
* `k` is a positive integer containing the number of clusters
* The cluster centroids should be initialized with a **multivariate uniform distribution** along each dimension in `d`:
	+ The minimum values for the distribution should be the minimum values of `X` along each dimension in `d`
	+ The maximum values for the distribution should be the maximum values of `X` along each dimension in `d`
	+ You should use `numpy.random.uniform` exactly once
* You are not allowed to use any loops
* Returns: a `numpy.ndarray` of shape `(k, d)` containing the initialized centroids for each cluster, or `None` on failure



```
alexa@ubuntu-xenial:0x01-clustering$ cat 0-main.py 
	#!/usr/bin/env python3
	
	import numpy as np
	import matplotlib.pyplot as plt
	initialize = __import__('0-initialize').initialize
	
	if __name__ == "__main__":
		np.random.seed(0)
		a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
		b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
		c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
		d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
		e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
		X = np.concatenate((a, b, c, d, e), axis=0)
		np.random.shuffle(X)
		plt.scatter(X[:, 0], X[:, 1], s=10)
		plt.show()
		print(initialize(X, 5))
	alexa@ubuntu-xenial:0x01-clustering$ ./0-main.py 
	
```

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/8/a5311b6ad0410a678aab.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T024523Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=fb07553e5bdc52ff35387fb9f024d8b1a39e5b15b5fdf91442e64ffcd7cbc852)



```
[[14.54730144 13.46780434]
	 [20.57098466 33.55245039]
	 [ 9.55556506 51.51143281]
	 [48.72458008 20.03154959]
	 [25.43826106 60.35542243]]
	alexa@ubuntu-xenial:0x01-clustering$
	
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `0-initialize.py`










 Help
 




×
#### Students who are done with "0\. Initialize K\-means"

















 Review your work
 




×
#### Correction of "0\. Initialize K\-means"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 0\. Initialize K\-means












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**6/6** 
pts









### 1\. K\-means




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def kmeans(X, k, iterations=1000):` that performs K\-means on a dataset:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset
	+ `n` is the number of data points
	+ `d` is the number of dimensions for each data point
* `k` is a positive integer containing the number of clusters
* `iterations` is a positive integer containing the maximum number of iterations that should be performed
* If no change in the cluster centroids occurs between iterations, your function should return
* Initialize the cluster centroids using a multivariate uniform distribution (based on`0-initialize.py`)
* If a cluster contains no data points during the update step, reinitialize its centroid
* You should use `numpy.random.uniform` exactly twice
* You may use at most 2 loops
* Returns: `C, clss`, or `None, None` on failure
	+ `C` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster
	+ `clss` is a `numpy.ndarray` of shape `(n,)` containing the index of the cluster in `C` that each data point belongs to



```
alexa@ubuntu-xenial:0x01-clustering$ cat 1-main.py 
	#!/usr/bin/env python3
	
	import numpy as np
	import matplotlib.pyplot as plt
	kmeans = __import__('1-kmeans').kmeans
	
	if __name__ == "__main__":
		np.random.seed(0)
		a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
		b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
		c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
		d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
		e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
		X = np.concatenate((a, b, c, d, e), axis=0)
		np.random.shuffle(X)
		C, clss = kmeans(X, 5)
		print(C)
		plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
		plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
		plt.show()
	alexa@ubuntu-xenial:0x01-clustering$ ./0-main.py 
	[[ 9.92511389 25.73098987]
	 [30.06722465 40.41123947]
	 [39.62770705 19.89843487]
	 [59.22766628 29.19796006]
	 [20.0835633  69.81592298]]
	
```

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/8/74c9c16e29333b39f7db.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T024523Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=752fdd43a0439f05ed0871076f1a1986a7edc4818654e72e09070b9df22013be)







**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `1-kmeans.py`










 Help
 




×
#### Students who are done with "1\. K\-means"

















 Review your work
 




×
#### Correction of "1\. K\-means"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 1\. K\-means












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**10/10** 
pts









### 2\. Variance




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def variance(X, C):` that calculates the total intra\-cluster variance for a data set:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
* `C` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster
* You are not allowed to use any loops
* Returns: `var`, or `None` on failure
	+ `var` is the total variance



```
alexa@ubuntu-xenial:0x01-clustering$ cat 2-main.py 
	#!/usr/bin/env python3
	
	import numpy as np
	kmeans = __import__('1-kmeans').kmeans
	variance = __import__('2-variance').variance
	
	if __name__ == "__main__":
		np.random.seed(0)
		a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
		b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
		c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
		d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
		e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
		X = np.concatenate((a, b, c, d, e), axis=0)
		np.random.shuffle(X)
	
		for k in range(1, 11):
			C, _ = kmeans(X, k)
			print('Variance with {} clusters: {}'.format(k, variance(X, C).round(5)))
	alexa@ubuntu-xenial:0x01-clustering$ ./2-main.py 
	Variance with 1 clusters: 157927.7052
	Variance with 2 clusters: 82095.68297
	Variance with 3 clusters: 34784.23723
	Variance with 4 clusters: 23158.40095
	Variance with 5 clusters: 7868.52123
	Variance with 6 clusters: 7406.93077
	Variance with 7 clusters: 6930.66361
	Variance with 8 clusters: 6162.15884
	Variance with 9 clusters: 5843.92455
	Variance with 10 clusters: 5727.41124
	alexa@ubuntu-xenial:0x01-clustering$ 
	
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `2-variance.py`










 Help
 




×
#### Students who are done with "2\. Variance"

















 Review your work
 




×
#### Correction of "2\. Variance"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 2\. Variance












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**6/6** 
pts









### 3\. Optimize k




 mandatory
 











 Score: 70\.00% (Checks completed: 70\.00%)
 


Write a function`def optimum_k(X, kmin=1, kmax=None, iterations=1000):` that tests for the optimum number of clusters by variance:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
* `kmin` is a positive integer containing the minimum number of clusters to check for (inclusive)
* `kmax` is a positive integer containing the maximum number of clusters to check for (inclusive)
* `iterations` is a positive integer containing the maximum number of iterations for K\-means
* This function should analyze *at least* 2 different cluster sizes
* You should use:
	+ `kmeans = __import__('1-kmeans').kmeans`
	+ `variance = __import__('2-variance').variance`
* You may use at most 2 loops
* Returns: `results, d_vars`, or `None, None` on failure
	+ `results` is a list containing the outputs of K\-means for each cluster size
	+ `d_vars` is a list containing the difference in variance from the smallest cluster size for each cluster size



```
alexa@ubuntu-xenial:0x01-clustering$ cat 3-main.py 
	#!/usr/bin/env python3
	
	import numpy as np
	import matplotlib.pyplot as plt
	optimum_k = __import__('3-optimum').optimum_k
	
	if __name__ == "__main__":
		np.random.seed(0)
		a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
		b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
		c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
		d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
		e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
		X = np.concatenate((a, b, c, d, e), axis=0)
		np.random.shuffle(X)
	
		results, d_vars = optimum_k(X, kmax=10)
		print(results)
		print(np.round(d_vars, 5))
		plt.scatter(list(range(1, 11)), d_vars)
		plt.xlabel('Clusters')
		plt.ylabel('Delta Variance')
		plt.title('Optimizing K-means')
		plt.show()
	alexa@ubuntu-xenial:0x01-clustering$ ./3-main.py 
	[(array([[31.78625503, 37.01090945]]), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0])), (array([[34.76990289, 28.71421162],
		   [20.14417812, 69.38429903]]), array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
		   1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
		   0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
		   0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0,
		   0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
		   1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
		   0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
		   1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
		   0, 1, 0, 0, 0, 0, 0, 0])), (array([[49.55185774, 24.76080087],
		   [20.0835633 , 69.81592298],
		   [19.8719982 , 32.85851127]]), array([2, 2, 1, 2, 0, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 2, 2, 0, 0, 1, 2, 2,
		   1, 0, 2, 2, 0, 0, 2, 0, 0, 2, 1, 2, 0, 2, 0, 2, 0, 1, 1, 0, 2, 2,
		   0, 2, 2, 2, 1, 0, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0,
		   2, 0, 1, 0, 1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 0, 2, 0, 1, 2, 0, 2, 2,
		   0, 1, 2, 2, 0, 2, 2, 1, 2, 1, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0,
		   2, 2, 0, 1, 0, 2, 2, 1, 2, 2, 0, 0, 2, 1, 0, 1, 1, 0, 1, 2, 2, 2,
		   0, 1, 0, 2, 2, 0, 2, 0, 2, 0, 1, 0, 1, 1, 0, 2, 2, 1, 0, 2, 0, 0,
		   1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 1, 1, 1, 1, 2, 0, 0, 1,
		   2, 0, 0, 2, 0, 0, 2, 1, 2, 1, 2, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0,
		   2, 2, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
		   1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0,
		   2, 1, 0, 2, 2, 0, 2, 0])), (array([[39.57566544, 20.48452248],
		   [20.0835633 , 69.81592298],
		   [19.62313956, 33.02895961],
		   [59.22766628, 29.19796006]]), array([2, 2, 1, 2, 3, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 2, 0, 3, 0, 1, 2, 2,
		   1, 3, 2, 2, 0, 0, 2, 3, 3, 2, 1, 2, 0, 2, 3, 2, 3, 1, 1, 0, 2, 2,
		   0, 2, 2, 2, 1, 3, 3, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3,
		   2, 0, 1, 3, 1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 3, 2, 3, 1, 2, 0, 2, 2,
		   3, 1, 2, 2, 3, 2, 2, 1, 2, 1, 0, 3, 2, 2, 3, 0, 0, 2, 2, 0, 3, 0,
		   2, 2, 0, 1, 0, 2, 2, 1, 2, 2, 0, 0, 2, 1, 0, 1, 1, 3, 1, 2, 2, 2,
		   3, 1, 3, 2, 2, 0, 2, 3, 2, 3, 1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 0,
		   1, 1, 1, 3, 3, 0, 2, 2, 2, 0, 0, 3, 0, 2, 1, 1, 1, 1, 2, 0, 3, 1,
		   2, 3, 3, 2, 0, 0, 2, 1, 2, 1, 0, 0, 1, 1, 3, 0, 3, 2, 0, 0, 3, 3,
		   2, 2, 3, 0, 0, 0, 3, 1, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
		   1, 0, 0, 2, 1, 0, 3, 0, 3, 1, 1, 0, 0, 3, 0, 3, 0, 2, 2, 3, 3, 3,
		   2, 1, 0, 2, 2, 0, 2, 0])), (array([[30.06722465, 40.41123947],
		   [59.22766628, 29.19796006],
		   [ 9.92511389, 25.73098987],
		   [20.0835633 , 69.81592298],
		   [39.62770705, 19.89843487]]), array([0, 0, 3, 0, 1, 2, 0, 4, 3, 4, 4, 0, 3, 4, 3, 2, 4, 1, 4, 3, 2, 0,
		   3, 1, 0, 2, 4, 4, 2, 1, 1, 0, 3, 2, 4, 0, 1, 0, 1, 3, 3, 4, 0, 2,
		   4, 0, 0, 0, 3, 1, 1, 3, 0, 3, 0, 2, 0, 2, 0, 0, 2, 2, 1, 2, 1, 1,
		   0, 4, 3, 1, 3, 4, 0, 0, 2, 0, 3, 2, 0, 3, 1, 0, 1, 3, 2, 4, 2, 0,
		   1, 3, 0, 2, 1, 0, 0, 3, 2, 3, 4, 1, 0, 2, 1, 4, 4, 0, 2, 4, 1, 4,
		   2, 2, 4, 3, 4, 2, 2, 3, 2, 2, 4, 4, 0, 3, 0, 3, 3, 1, 3, 2, 0, 2,
		   1, 3, 1, 2, 0, 4, 0, 1, 0, 1, 3, 1, 3, 3, 1, 2, 2, 3, 1, 2, 1, 4,
		   3, 3, 3, 1, 1, 4, 2, 2, 0, 4, 4, 1, 4, 2, 3, 3, 3, 3, 0, 4, 1, 3,
		   2, 1, 1, 2, 4, 4, 2, 3, 2, 3, 4, 4, 3, 3, 1, 4, 1, 0, 4, 4, 1, 1,
		   2, 0, 1, 4, 4, 0, 1, 3, 1, 2, 0, 1, 0, 2, 0, 2, 2, 0, 0, 2, 3, 0,
		   3, 4, 4, 2, 3, 4, 1, 4, 1, 3, 3, 4, 4, 1, 4, 1, 4, 0, 2, 1, 1, 1,
		   2, 3, 4, 0, 2, 4, 2, 4])), (array([[44.18492017, 16.98881789],
		   [30.06722465, 40.41123947],
		   [38.18858711, 20.81726128],
		   [59.22766628, 29.19796006],
		   [20.0835633 , 69.81592298],
		   [ 9.92511389, 25.73098987]]), array([1, 1, 4, 1, 3, 5, 1, 2, 4, 0, 2, 1, 4, 2, 4, 5, 2, 3, 2, 4, 5, 1,
		   4, 3, 1, 5, 2, 2, 5, 3, 3, 1, 4, 5, 2, 1, 3, 1, 3, 4, 4, 2, 1, 5,
		   0, 1, 1, 1, 4, 3, 3, 4, 1, 4, 1, 5, 1, 5, 1, 1, 5, 5, 3, 5, 3, 3,
		   1, 2, 4, 3, 4, 2, 1, 1, 5, 1, 4, 5, 1, 4, 3, 1, 3, 4, 5, 2, 5, 1,
		   3, 4, 1, 5, 3, 1, 1, 4, 5, 4, 2, 3, 1, 5, 3, 2, 2, 1, 5, 2, 3, 2,
		   5, 5, 2, 4, 2, 5, 5, 4, 5, 5, 0, 2, 1, 4, 1, 4, 4, 3, 4, 5, 1, 5,
		   3, 4, 3, 5, 1, 0, 1, 3, 1, 3, 4, 3, 4, 4, 3, 5, 5, 4, 3, 5, 3, 0,
		   4, 4, 4, 3, 3, 0, 5, 5, 1, 0, 2, 3, 2, 5, 4, 4, 4, 4, 1, 0, 3, 4,
		   5, 3, 3, 5, 2, 2, 5, 4, 5, 4, 2, 2, 4, 4, 3, 0, 3, 1, 0, 2, 3, 3,
		   5, 1, 3, 2, 2, 1, 3, 4, 3, 5, 1, 3, 1, 5, 1, 5, 5, 1, 1, 5, 4, 1,
		   4, 2, 2, 5, 4, 2, 3, 2, 3, 4, 4, 0, 2, 3, 0, 3, 2, 1, 5, 3, 3, 3,
		   5, 4, 2, 1, 5, 2, 5, 2])), (array([[26.23935735, 39.56231098],
		   [33.86397001, 36.21416257],
		   [32.10246392, 43.52452575],
		   [39.78587939, 19.72783208],
		   [59.22766628, 29.19796006],
		   [20.0835633 , 69.81592298],
		   [ 9.92511389, 25.73098987]]), array([0, 1, 5, 2, 4, 6, 1, 3, 5, 3, 3, 2, 5, 3, 5, 6, 3, 4, 3, 5, 6, 0,
		   5, 4, 1, 6, 3, 3, 6, 4, 4, 0, 5, 6, 3, 2, 4, 1, 4, 5, 5, 3, 2, 6,
		   3, 0, 0, 1, 5, 4, 4, 5, 2, 5, 0, 6, 2, 6, 2, 1, 6, 6, 4, 6, 4, 4,
		   0, 3, 5, 4, 5, 3, 0, 2, 6, 1, 5, 6, 2, 5, 4, 2, 4, 5, 6, 3, 6, 0,
		   4, 5, 0, 6, 4, 2, 0, 5, 6, 5, 3, 4, 2, 6, 4, 3, 3, 0, 6, 3, 4, 3,
		   6, 6, 3, 5, 3, 6, 6, 5, 6, 6, 3, 3, 0, 5, 1, 5, 5, 4, 5, 6, 0, 6,
		   4, 5, 4, 6, 0, 3, 0, 4, 2, 4, 5, 4, 5, 5, 4, 6, 6, 5, 4, 6, 4, 3,
		   5, 5, 5, 4, 4, 3, 6, 6, 2, 3, 3, 4, 3, 6, 5, 5, 5, 5, 0, 3, 4, 5,
		   6, 4, 4, 6, 3, 3, 6, 5, 6, 5, 1, 3, 5, 5, 4, 3, 4, 2, 3, 3, 4, 4,
		   6, 2, 4, 3, 3, 1, 4, 5, 4, 6, 1, 4, 0, 6, 0, 6, 6, 2, 2, 6, 5, 0,
		   5, 3, 3, 6, 5, 3, 4, 3, 4, 5, 5, 3, 3, 4, 3, 4, 3, 0, 6, 4, 4, 4,
		   6, 5, 3, 1, 6, 3, 6, 3])), (array([[46.45917139, 23.16813158],
		   [17.46350686, 68.10438494],
		   [59.60504708, 29.19922401],
		   [32.37545229, 35.93762723],
		   [38.66481538, 19.34921236],
		   [28.96634723, 42.29059653],
		   [ 9.92511389, 25.73098987],
		   [24.01364794, 72.38323004]]), array([5, 3, 1, 5, 0, 6, 3, 4, 7, 4, 4, 5, 1, 4, 1, 6, 4, 2, 4, 1, 6, 5,
		   7, 2, 3, 6, 4, 4, 6, 2, 2, 3, 7, 6, 4, 3, 2, 3, 2, 1, 7, 4, 5, 6,
		   0, 5, 5, 3, 7, 2, 2, 1, 5, 7, 5, 6, 5, 6, 5, 3, 6, 6, 2, 6, 2, 2,
		   5, 4, 1, 2, 1, 4, 5, 5, 6, 3, 7, 6, 5, 1, 2, 5, 2, 7, 6, 0, 6, 5,
		   2, 1, 3, 6, 2, 5, 5, 7, 6, 7, 4, 2, 5, 6, 2, 4, 0, 5, 6, 4, 2, 4,
		   6, 6, 4, 1, 4, 6, 6, 7, 6, 6, 0, 4, 5, 7, 3, 7, 1, 2, 1, 6, 3, 6,
		   2, 1, 2, 6, 5, 4, 5, 2, 5, 2, 7, 2, 1, 7, 2, 6, 6, 1, 2, 6, 2, 4,
		   7, 1, 1, 2, 2, 0, 6, 6, 5, 0, 4, 2, 0, 6, 7, 1, 1, 7, 5, 4, 2, 7,
		   6, 2, 0, 6, 4, 4, 6, 1, 6, 1, 3, 4, 7, 1, 2, 4, 2, 5, 4, 4, 2, 2,
		   6, 5, 2, 4, 4, 3, 2, 1, 2, 6, 3, 2, 3, 6, 5, 6, 6, 5, 5, 6, 1, 5,
		   1, 4, 4, 6, 1, 4, 2, 4, 2, 1, 1, 0, 4, 2, 4, 2, 4, 5, 6, 2, 2, 2,
		   6, 1, 4, 3, 6, 4, 6, 4])), (array([[41.41465903, 21.62570842],
		   [37.61417321, 17.19733029],
		   [26.6124459 , 43.91604964],
		   [16.74238916, 69.20086704],
		   [ 9.92511389, 25.73098987],
		   [24.00581119, 70.53794517],
		   [31.42335817, 38.75662838],
		   [55.10816891, 27.75253127],
		   [61.16625328, 29.87816185]]), array([6, 6, 3, 6, 7, 4, 6, 1, 5, 0, 1, 2, 3, 1, 3, 4, 1, 7, 0, 3, 4, 2,
		   3, 7, 6, 4, 1, 0, 4, 8, 8, 6, 5, 4, 1, 6, 8, 6, 7, 3, 5, 0, 2, 4,
		   0, 2, 6, 6, 5, 8, 7, 3, 2, 5, 6, 4, 6, 4, 6, 6, 4, 4, 8, 4, 8, 8,
		   2, 1, 3, 8, 3, 0, 6, 6, 4, 6, 5, 4, 6, 3, 7, 6, 8, 5, 4, 0, 4, 2,
		   7, 3, 6, 4, 7, 6, 6, 5, 4, 5, 0, 8, 6, 4, 8, 0, 0, 6, 4, 0, 8, 1,
		   4, 4, 0, 3, 0, 4, 4, 5, 4, 4, 0, 0, 2, 5, 6, 5, 3, 8, 3, 4, 6, 4,
		   8, 3, 7, 4, 2, 0, 2, 7, 2, 8, 5, 8, 3, 5, 8, 4, 4, 3, 7, 4, 7, 0,
		   5, 3, 5, 8, 7, 0, 4, 4, 6, 0, 1, 8, 0, 4, 5, 3, 3, 5, 6, 1, 8, 5,
		   4, 8, 7, 4, 1, 0, 4, 5, 4, 3, 6, 0, 5, 3, 8, 1, 8, 6, 1, 1, 8, 8,
		   4, 6, 8, 1, 0, 6, 8, 3, 8, 4, 6, 8, 6, 4, 6, 4, 4, 6, 2, 4, 5, 2,
		   3, 1, 0, 4, 3, 0, 7, 1, 8, 5, 3, 0, 1, 8, 1, 8, 0, 2, 4, 7, 8, 8,
		   4, 3, 0, 6, 4, 1, 4, 1])), (array([[22.16577861, 70.20228452],
		   [26.6124459 , 43.91604964],
		   [19.74208872, 64.1000882 ],
		   [39.78587939, 19.72783208],
		   [15.57193046, 70.58516939],
		   [60.58250295, 27.97262624],
		   [25.38793378, 73.72176884],
		   [ 9.92511389, 25.73098987],
		   [55.37159269, 32.68544864],
		   [31.42335817, 38.75662838]]), array([9, 9, 4, 9, 8, 7, 9, 3, 0, 3, 3, 1, 4, 3, 0, 7, 3, 8, 3, 0, 7, 1,
		   4, 5, 9, 7, 3, 3, 7, 5, 5, 9, 0, 7, 3, 9, 5, 9, 5, 2, 0, 3, 1, 7,
		   3, 1, 9, 9, 6, 5, 5, 4, 1, 6, 9, 7, 9, 7, 9, 9, 7, 7, 8, 7, 5, 5,
		   1, 3, 2, 5, 0, 3, 9, 9, 7, 9, 0, 7, 9, 2, 8, 9, 5, 6, 7, 3, 7, 1,
		   5, 4, 9, 7, 8, 9, 9, 0, 7, 6, 3, 5, 9, 7, 5, 3, 3, 9, 7, 3, 5, 3,
		   7, 7, 3, 4, 3, 7, 7, 6, 7, 7, 3, 3, 1, 0, 9, 0, 4, 8, 4, 7, 9, 7,
		   5, 4, 5, 7, 1, 3, 1, 8, 1, 8, 6, 5, 4, 0, 8, 7, 7, 4, 5, 7, 8, 3,
		   6, 2, 2, 5, 5, 3, 7, 7, 9, 3, 3, 5, 3, 7, 6, 4, 2, 6, 9, 3, 5, 0,
		   7, 5, 8, 7, 3, 3, 7, 0, 7, 4, 9, 3, 6, 2, 5, 3, 8, 9, 3, 3, 5, 8,
		   7, 9, 5, 3, 3, 9, 5, 4, 5, 7, 9, 5, 9, 7, 9, 7, 7, 9, 1, 7, 2, 1,
		   4, 3, 3, 7, 4, 3, 5, 3, 5, 2, 4, 3, 3, 5, 3, 5, 3, 1, 7, 5, 5, 5,
		   7, 2, 3, 9, 7, 3, 7, 3]))]
	[     0.       75832.02223 123143.46797 134769.30424 150059.18397
	 150520.77443 150997.04159 151765.54636 152083.78065 152200.29395]
	
```

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/8/4cd46d5d9463dad96a37.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T024523Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d737cc4ea9db285b9db974ecc446439222f2dc3c217e2b626fffe40637e44b2e)







**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `3-optimum.py`










 Help
 




×
#### Students who are done with "3\. Optimize k"

















 Review your work
 




×
#### Correction of "3\. Optimize k"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 3\. Optimize k












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**7/10** 
pts









### 4\. Initialize GMM




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def initialize(X, k):` that initializes variables for a Gaussian Mixture Model:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
* `k` is a positive integer containing the number of clusters
* You are not allowed to use any loops
* Returns: `pi, m, S`, or `None, None, None` on failure
	+ `pi` is a `numpy.ndarray` of shape `(k,)` containing the priors for each cluster, initialized evenly
	+ `m` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster, initialized with K\-means
	+ `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the covariance matrices for each cluster, initialized as identity matrices
* You should use `kmeans = __import__('1-kmeans').kmeans`



```
alexa@ubuntu-xenial:0x01-clustering$ cat 4-main.py 
	#!/usr/bin/env python3
	
	import numpy as np
	initialize = __import__('4-initialize').initialize
	
	if __name__ == '__main__':
		np.random.seed(11)
		a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
		b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
		c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
		d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
		X = np.concatenate((a, b, c, d), axis=0)
		np.random.shuffle(X)
		pi, m, S = initialize(X, 4)
		print(pi)
		print(m)
		print(S)
	alexa@ubuntu-xenial:0x01-clustering$ ./4-main.py 
	[0.25 0.25 0.25 0.25]
	[[54.73711515 31.81393242]
	 [16.84012557 31.20248225]
	 [21.43215816 65.50449077]
	 [32.3301925  41.80664127]]
	[[[1. 0.]
	  [0. 1.]]
	
	 [[1. 0.]
	  [0. 1.]]
	
	 [[1. 0.]
	  [0. 1.]]
	
	 [[1. 0.]
	  [0. 1.]]]
	alexa@ubuntu-xenial:0x01-clustering$
	
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `4-initialize.py`










 Help
 




×
#### Students who are done with "4\. Initialize GMM"

















 Review your work
 




×
#### Correction of "4\. Initialize GMM"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 4\. Initialize GMM












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**6/6** 
pts









### 5\. PDF




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def pdf(X, m, S):` that calculates the probability density function of a Gaussian distribution:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the data points whose PDF should be evaluated
* `m` is a `numpy.ndarray` of shape `(d,)` containing the mean of the distribution
* `S` is a `numpy.ndarray` of shape `(d, d)` containing the covariance of the distribution
* You are not allowed to use any loops
* You are not allowed to use the function `numpy.diag` or the method `numpy.ndarray.diagonal`
* Returns: `P`, or `None` on failure
	+ `P` is a `numpy.ndarray` of shape `(n,)` containing the PDF values for each data point
* All values in `P` should have a minimum value of `1e-300`



```
alexa@ubuntu-xenial:0x01-clustering$ cat 5-main.py
	#!/usr/bin/env python3
	
	import numpy as np
	pdf = __import__('5-pdf').pdf
	
	if __name__ == '__main__':
		np.random.seed(0)
		m = np.array([12, 30, 10])
		S = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
		X = np.random.multivariate_normal(m, S, 10000)
		P = pdf(X, m, S)
		print(P)
	alexa@ubuntu-xenial:0x01-clustering$ ./5-main.py
	[3.47450910e-05 2.53649178e-06 1.80348301e-04 ... 1.24604061e-04
	 1.86345129e-04 2.59397003e-05]
	alexa@ubuntu-xenial:0x01-clustering$
	
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `5-pdf.py`










 Help
 




×
#### Students who are done with "5\. PDF"

















 Review your work
 




×
#### Correction of "5\. PDF"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 5\. PDF












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**8/8** 
pts









### 6\. Expectation




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def expectation(X, pi, m, S):` that calculates the expectation step in the EM algorithm for a GMM:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
* `pi` is a `numpy.ndarray` of shape `(k,)` containing the priors for each cluster
* `m` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster
* `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the covariance matrices for each cluster
* You may use at most 1 loop
* Returns: `g, l`, or `None, None` on failure
	+ `g` is a `numpy.ndarray` of shape `(k, n)` containing the posterior probabilities for each data point in each cluster
	+ `l` is the total log likelihood
* You should use `pdf = __import__('5-pdf').pdf`



```
alexa@ubuntu-xenial:0x01-clustering$ cat 6-main.py
	#!/usr/bin/env python3
	
	import numpy as np
	initialize = __import__('4-initialize').initialize
	expectation = __import__('6-expectation').expectation
	
	if __name__ == '__main__':
		np.random.seed(11)
		a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
		b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
		c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
		d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
		X = np.concatenate((a, b, c, d), axis=0)
		np.random.shuffle(X)
		pi, m, S = initialize(X, 4)
		g, l = expectation(X, pi, m, S)
		print(g)
		print(np.sum(g, axis=0))
		print(l)
	alexa@ubuntu-xenial:0x01-clustering$ ./6-main.py
	[[1.98542668e-055 1.00000000e+000 1.56526421e-185 ... 1.00000000e+000
	  3.70567311e-236 1.91892348e-012]
	 [6.97883333e-085 2.28658376e-279 9.28518983e-065 ... 8.12227631e-287
	  1.53690661e-032 3.17417182e-181]
	 [9.79811365e-234 2.28658376e-279 2.35073465e-095 ... 1.65904890e-298
	  9.62514613e-068 5.67072057e-183]
	 [1.00000000e+000 7.21133039e-186 1.00000000e+000 ... 2.42138447e-125
	  1.00000000e+000 1.00000000e+000]]
	[1. 1. 1. ... 1. 1. 1.]
	-652797.7866541843
	alexa@ubuntu-xenial:0x01-clustering$
	
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `6-expectation.py`










 Help
 




×
#### Students who are done with "6\. Expectation"

















 Review your work
 




×
#### Correction of "6\. Expectation"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 6\. Expectation












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**8/8** 
pts









### 7\. Maximization




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def maximization(X, g):` that calculates the maximization step in the EM algorithm for a GMM:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
* `g` is a `numpy.ndarray` of shape `(k, n)` containing the posterior probabilities for each data point in each cluster
* You may use at most 1 loop
* Returns: `pi, m, S`, or `None, None, None` on failure
	+ `pi` is a `numpy.ndarray` of shape `(k,)` containing the updated priors for each cluster
	+ `m` is a `numpy.ndarray` of shape `(k, d)` containing the updated centroid means for each cluster
	+ `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the updated covariance matrices for each cluster



```
alexa@ubuntu-xenial:0x01-clustering$ cat 7-main.py
	#!/usr/bin/env python3
	
	import numpy as np
	initialize = __import__('4-initialize').initialize
	expectation = __import__('6-expectation').expectation
	maximization = __import__('7-maximization').maximization
	
	if __name__ == '__main__':
		np.random.seed(11)
		a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
		b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
		c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
		d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
		X = np.concatenate((a, b, c, d), axis=0)
		np.random.shuffle(X)
		pi, m, S = initialize(X, 4)
		g, _ = expectation(X, pi, m, S)
		pi, m, S = maximization(X, g)
		print(pi)
		print(m)
		print(S)
	alexa@ubuntu-xenial:0x01-clustering$ ./7-main.py
	[0.10104901 0.24748822 0.1193333  0.53212947]
	[[54.7440558  31.80888393]
	 [16.84099873 31.20560148]
	 [21.42588061 65.51441875]
	 [32.33208369 41.80830251]]
	[[[64.05063663 -2.13941814]
	  [-2.13941814 41.90354928]]
	
	 [[72.72404579  9.96322554]
	  [ 9.96322554 53.05035303]]
	
	 [[46.20933259  1.08979413]
	  [ 1.08979413 66.9841323 ]]
	
	 [[35.04054823 -0.94790014]
	  [-0.94790014 45.14948772]]]
	alexa@ubuntu-xenial:0x01-clustering$
	
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `7-maximization.py`










 Help
 




×
#### Students who are done with "7\. Maximization"

















 Review your work
 




×
#### Correction of "7\. Maximization"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 7\. Maximization












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**6/6** 
pts









### 8\. EM




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):` that performs the expectation maximization for a GMM:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
* `k` is a positive integer containing the number of clusters
* `iterations` is a positive integer containing the maximum number of iterations for the algorithm
* `tol` is a non\-negative float containing tolerance of the log likelihood, used to determine early stopping i.e. if the difference is less than or equal to `tol` you should stop the algorithm
* `verbose` is a boolean that determines if you should print information about the algorithm
	+ If `True`, print `Log Likelihood after {i} iterations: {l}` every 10 iterations and after the last iteration
	+ `{i}` is the number of iterations of the EM algorithm
	+ `{l}` is the log likelihood, rounded to 5 decimal places
* You should use:
	+ `initialize = __import__('4-initialize').initialize`
	+ `expectation = __import__('6-expectation').expectation`
	+ `maximization = __import__('7-maximization').maximization`
* You may use at most 1 loop
* Returns: `pi, m, S, g, l`, or `None, None, None, None, None` on failure
	+ `pi` is a `numpy.ndarray` of shape `(k,)` containing the priors for each cluster
	+ `m` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster
	+ `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the covariance matrices for each cluster
	+ `g` is a `numpy.ndarray` of shape `(k, n)` containing the probabilities for each data point in each cluster
	+ `l` is the log likelihood of the model



```
alexa@ubuntu-xenial:0x01-clustering$ cat 8-main.py
	#!/usr/bin/env python3
	
	import matplotlib.pyplot as plt
	import numpy as np
	expectation_maximization = __import__('8-EM').expectation_maximization
	
	if __name__ == '__main__':
		np.random.seed(11)
		a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
		b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
		c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
		d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
		X = np.concatenate((a, b, c, d), axis=0)
		np.random.shuffle(X)
		k = 4
		pi, m, S, g, l = expectation_maximization(X, k, 150, verbose=True)
		clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)
		plt.scatter(X[:, 0], X[:, 1], s=20, c=clss)
		plt.scatter(m[:, 0], m[:, 1], s=50, c=np.arange(k), marker='*')
		plt.show()
		print(X.shape[0] * pi)
		print(m)
		print(S)
		print(l)
	alexa@ubuntu-xenial:0x01-clustering$ ./8-main.py
	Log Likelihood after 0 iterations: -652797.78665
	Log Likelihood after 10 iterations: -94855.45662
	Log Likelihood after 20 iterations: -94714.52057
	Log Likelihood after 30 iterations: -94590.87362
	Log Likelihood after 40 iterations: -94440.40559
	Log Likelihood after 50 iterations: -94439.93891
	Log Likelihood after 52 iterations: -94439.93889
	
```

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/11/9a6a49a5317fa046854f.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T024523Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=0baa6ebfda5ff42fe80ebb3a1bc7cabf395fc809ef5f7431266c8e6e59020be7)



```
[ 761.03239903  747.62391034 1005.60275934 9985.74093129]
	[[60.18888335 30.19707607]
	 [ 5.05794926 24.92588821]
	 [20.03438453 69.84721009]
	 [29.89607379 40.12519148]]
	[[[16.85183426  0.2547388 ]
	  [ 0.2547388  16.49432111]]
	
	 [[15.19555672  9.62661086]
	  [ 9.62661086 15.47295413]]
	
	 [[35.58332494 11.08419454]
	  [11.08419454 33.09463207]]
	
	 [[74.52083678  5.20755533]
	  [ 5.20755533 73.87299705]]]
	-94439.93889004056
	alexa@ubuntu-xenial:0x01-clustering$
	
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `8-EM.py`










 Help
 




×
#### Students who are done with "8\. EM"

















 Review your work
 




×
#### Correction of "8\. EM"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 8\. EM












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**12/12** 
pts









### 9\. BIC




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):` that finds the best number of clusters for a GMM using the Bayesian Information Criterion:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
* `kmin` is a positive integer containing the minimum number of clusters to check for (inclusive)
* `kmax` is a positive integer containing the maximum number of clusters to check for (inclusive)
	+ If `kmax` is `None`, `kmax` should be set to the maximum number of clusters possible
* `iterations` is a positive integer containing the maximum number of iterations for the EM algorithm
* `tol` is a non\-negative float containing the tolerance for the EM algorithm
* `verbose` is a boolean that determines if the EM algorithm should print information to the standard output
* You should use `expectation_maximization = __import__('8-EM').expectation_maximization`
* You may use at most 1 loop
* Returns: `best_k, best_result, l, b`, or `None, None, None, None` on failure
	+ `best_k` is the best value for k based on its BIC
	+ `best_result` is tuple containing `pi, m, S`
		- `pi` is a `numpy.ndarray` of shape `(k,)` containing the cluster priors for the best number of clusters
		- `m` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for the best number of clusters
		- `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the covariance matrices for the best number of clusters
	+ `l` is a `numpy.ndarray` of shape `(kmax - kmin + 1)` containing the log likelihood for each cluster size tested
	+ `b` is a `numpy.ndarray` of shape `(kmax - kmin + 1)` containing the BIC value for each cluster size tested
		- Use: `BIC = p * ln(n) - 2 * l`
		- `p` is the number of parameters required for the model
		- `n` is the number of data points used to create the model
		- `l` is the log likelihood of the model



```
alexa@ubuntu-xenial:0x01-clustering$ cat 9-main.py
	#!/usr/bin/env python3
	
	import matplotlib.pyplot as plt
	import numpy as np
	BIC = __import__('9-BIC').BIC
	
	if __name__ == '__main__':
		np.random.seed(11)
		a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
		b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
		c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
		d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
		X = np.concatenate((a, b, c, d), axis=0)
		np.random.shuffle(X)
		best_k, best_result, l, b = BIC(X, kmin=1, kmax=10)
		print(best_k)
		print(best_result)
		print(l)
		print(b)
		x = np.arange(1, 11)
		plt.plot(x, l, 'r')
		plt.xlabel('Clusters')
		plt.ylabel('Log Likelihood')
		plt.tight_layout()
		plt.show()
		plt.plot(x, b, 'b')
		plt.xlabel('Clusters')
		plt.ylabel('BIC')
		plt.tight_layout()
		plt.show()
	alexa@ubuntu-xenial:0x01-clustering$ ./9-main.py
	4
	(array([0.79885962, 0.08044842, 0.06088258, 0.05980938]), array([[29.89606417, 40.12518027],
		   [20.0343883 , 69.84718588],
		   [60.18888407, 30.19707372],
		   [ 5.05788987, 24.92583792]]), array([[[74.52101284,  5.20770764],
			[ 5.20770764, 73.8729309 ]],
	
		   [[35.58334497, 11.08416742],
			[11.08416742, 33.09483747]],
	
		   [[16.85183256,  0.25475122],
			[ 0.25475122, 16.4943092 ]],
	
		   [[15.19520213,  9.62633552],
			[ 9.62633552, 15.47268905]]]))
	[-98801.40298366 -96729.95558846 -95798.40406023 -94439.93888882
	 -94435.87750008 -94428.62217176 -94426.71159745 -94425.5860871
	 -94421.41864281 -94416.43390835]
	[197649.97338694 193563.67950008 191757.17734716 189096.84790787
	 189145.32603394 189187.41628084 189240.19603576 189294.54591859
	 189342.81193356 189389.44336818]
	
```

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/8/3ce66d375c09b61bf6b4.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T024523Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d03ee4409d63fc06be071770073e426b8a580b08c13873630f371a854090492a)


![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/8/7434fa918f1cff45a2f9.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T024523Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=a8cc10e9e401569d31c7493920ff64c78841641c00c53e8175c87e757d3babbc)







**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `9-BIC.py`










 Help
 




×
#### Students who are done with "9\. BIC"

















 Review your work
 




×
#### Correction of "9\. BIC"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 9\. BIC












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**14/14** 
pts









### 10\. Hello, sklearn!




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def kmeans(X, k):` that performs K\-means on a dataset:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset
* `k` is the number of clusters
* The only import you are allowed to use is `import sklearn.cluster`
* Returns: `C, clss`
	+ `C` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster
	+ `clss` is a `numpy.ndarray` of shape `(n,)` containing the index of the cluster in `C` that each data point belongs to



```
alexa@ubuntu-xenial:0x01-clustering$ cat 10-main.py 
	#!/usr/bin/env python3
	import numpy as np
	import matplotlib.pyplot as plt
	kmeans = __import__('10-kmeans').kmeans
	
	if __name__ == "__main__":
		np.random.seed(0)
		a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
		b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
		c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
		d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
		e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
		X = np.concatenate((a, b, c, d, e), axis=0)
		np.random.shuffle(X)
	
		C, clss = kmeans(X, 5)
		print(C)
		plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
		plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
		plt.show()
	alexa@ubuntu-xenial:0x01-clustering$ ./10-main.py 
	[[30.06722465 40.41123947]
	 [59.22766628 29.19796006]
	 [ 9.92511389 25.73098987]
	 [20.0835633  69.81592298]
	 [39.62770705 19.89843487]]
	
```

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/8/9ed61d0ec495095f80bf.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T024523Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f0b2ca20f30aeda8336b3c35a20e00e10212168d497b090d2ac416af386488eb)







**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `10-kmeans.py`










 Help
 




×
#### Students who are done with "10\. Hello, sklearn!"

















 Review your work
 




×
#### Correction of "10\. Hello, sklearn!"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 10\. Hello, sklearn!












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**4/4** 
pts









### 11\. GMM




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def gmm(X, k):` that calculates a GMM from a dataset:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset
* `k` is the number of clusters
* The only import you are allowed to use is `import sklearn.mixture`
* Returns: `pi, m, S, clss, bic`
	+ `pi` is a `numpy.ndarray` of shape `(k,)` containing the cluster priors
	+ `m` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means
	+ `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the covariance matrices
	+ `clss` is a `numpy.ndarray` of shape `(n,)` containing the cluster indices for each data point
	+ `bic` is a `numpy.ndarray` of shape `(kmax - kmin + 1)` containing the BIC value for each cluster size tested



```
alexa@ubuntu-xenial:0x01-clustering$ cat 11-main.py 
	#!/usr/bin/env python3
	import matplotlib.pyplot as plt
	import numpy as np
	gmm = __import__('11-gmm').gmm
	
	if __name__ == '__main__':
		np.random.seed(11)
		a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
		b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
		c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
		d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
		X = np.concatenate((a, b, c, d), axis=0)
		np.random.shuffle(X)
	
		pi, m, S, clss, bic = gmm(X, 4)
		print(pi)
		print(m)
		print(S)
		print(bic)
		plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
		plt.scatter(m[:, 0], m[:, 1], s=50, marker='*', c=list(range(4)))
		plt.show()
	alexa@ubuntu-xenial:0x01-clustering$ ./11-main.py 
	[0.68235847 0.17835524 0.06079717 0.07848911]
	[[30.6032058  40.76046968]
	 [18.77413861 32.82638046]
	 [60.2071779  30.24071437]
	 [20.00542639 70.02497115]]
	[[[ 6.96776628e+01 -3.66204575e+00]
	  [-3.66204575e+00  7.82087329e+01]]
	
	 [[ 1.54935691e+02  8.17658780e+01]
	  [ 8.17658780e+01  6.56306521e+01]]
	
	 [[ 1.66980918e+01  1.00778176e-02]
	  [ 1.00778176e-02  1.66498476e+01]]
	
	 [[ 3.55102735e+01  1.13250004e+01]
	  [ 1.13250004e+01  3.21335458e+01]]]
	189794.11897876553
	
```

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/8/4d20ec068e424c587a98.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T024523Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2fcb4d0e85449760b9cd9e26590b0314263a2d664fc9c11f3fc3822852c84e7f)







**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `11-gmm.py`










 Help
 




×
#### Students who are done with "11\. GMM"

















 Review your work
 




×
#### Correction of "11\. GMM"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 11\. GMM












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**4/4** 
pts









### 12\. Agglomerative




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write a function `def agglomerative(X, dist):` that performs agglomerative clustering on a dataset:


* `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset
* `dist` is the maximum cophenetic distance for all clusters
* Performs agglomerative clustering with Ward linkage
* Displays the dendrogram with each cluster displayed in a different color
* The only imports you are allowed to use are:
	+ `import scipy.cluster.hierarchy`
	+ `import matplotlib.pyplot as plt`
* Returns: `clss`, a `numpy.ndarray` of shape `(n,)` containing the cluster indices for each data point



```
alexa@ubuntu-xenial:0x01-clustering$ cat 12-main.py 
	#!/usr/bin/env python3
	import matplotlib.pyplot as plt
	import numpy as np
	agglomerative = __import__('12-agglomerative').agglomerative
	
	if __name__ == '__main__':
		np.random.seed(0)
		a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
		b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
		c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
		d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=100)
		e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=100)
		X = np.concatenate((a, b, c, d, e), axis=0)
		np.random.shuffle(X)
	
		clss = agglomerative(X, 100)
		plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
		plt.show()
	alexa@ubuntu-xenial:0x01-clustering$ ./12-main.py 
	
```

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/8/06cc78d5e161cc719bdd.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T024523Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=fbf928e8ef0f64586a3aa85054e6d99faca28f4475f7f4e1e3e8894031d4b8f1)


![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/8/a1dffaac9991e9518a8b.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T024523Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=6b6fc90f8d110e81ae4d2de59016a1e669d94268c80fc406346e249cf0fd7680)







**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `unsupervised_learning/clustering`
* File: `12-agglomerative.py`










 Help
 




×
#### Students who are done with "12\. Agglomerative"

















 QA Review
 




×
#### 12\. Agglomerative












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**6/6** 
pts









[Previous project](/projects/2329)  



 Next project 







### Score

![Project badge](/assets/pathway/004_color-d2fbcfb42ba7481834896ecc89a6f0ae65762b3c1096691dd0f820f7f29e3389.png)97%Good job! You can still grab some points!

Now that you are ready to be reviewed, share your link to your peers. You can find some [here](#available-reviewers-modal).

×#### Contact one of your peers

https://intranet.hbtn.io/corrections/853580/correctDon't forget to [review one of them](/corrections/to_review).

Next project: Hidden Markov Models

[Open the next project](/projects/2331)


### Score

![Project badge](/assets/pathway/004_color-d2fbcfb42ba7481834896ecc89a6f0ae65762b3c1096691dd0f820f7f29e3389.png)97%Good job! You can still grab some points!

Now that you are ready to be reviewed, share your link to your peers. You can find some [here](#available-reviewers-modal).

×#### Contact one of your peers

https://intranet.hbtn.io/corrections/853580/correctDon't forget to [review one of them](/corrections/to_review).

Next project: Hidden Markov Models

[Open the next project](/projects/2331)



### Tasks list




* [Mandatory](#mandatory)
* [Advanced](#advanced)





0\. `Initialize K-means`
**100\.00%**


1\. `K-means`
**100\.00%**


2\. `Variance`
**100\.00%**


3\. `Optimize k`
**70\.00%**


4\. `Initialize GMM`
**100\.00%**


5\. `PDF`
**100\.00%**


6\. `Expectation`
**100\.00%**


7\. `Maximization`
**100\.00%**


8\. `EM`
**100\.00%**


9\. `BIC`
**100\.00%**


10\. `Hello, sklearn!`
**100\.00%**


11\. `GMM`
**100\.00%**


12\. `Agglomerative`
**100\.00%**









×#### Recommended Sandboxes

New sandbox * 
* US East (N. Virginia)
* Ubuntu 18\.04
* Ubuntu 22\.04
* 
* South America (São Paulo)
* Ubuntu 18\.04
* Ubuntu 22\.04
* 
* Europe (Paris)
* Ubuntu 18\.04
* Ubuntu 22\.04
* 
* Asia Pacific (Sydney)
* Ubuntu 18\.04
* Ubuntu 22\.04
No sandboxes yet!





