# Advanced Linear Algebra
## Resources
*   [The determinant | Essence of linear algebra](https://www.youtube.com/watch?v=Ip3X9LOh2dk&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=8 "The determinant | Essence of linear algebra")
*   [Determinant of a Matrix](https://www.mathsisfun.com/algebra/matrix-determinant.html "Determinant of a Matrix")
*   [Determinant](https://mathworld.wolfram.com/Determinant.html "Determinant")
*   [Determinant of an empty matrix](https://www.quora.com/What-is-the-determinant-of-an-empty-matrix-such-as-a-0x0-matrix "Determinant of an empty matrix")
*   [Inverse matrices, column space and null space](https://www.youtube.com/watch?v=uQhTuRlWMxw&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=9 "Inverse matrices, column space and null space")
*   [Inverse of a Matrix using Minors, Cofactors and Adjugate](https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html "Inverse of a Matrix using Minors, Cofactors and Adjugate")
*   [Minor](https://mathworld.wolfram.com/Minor.html "Minor")
*   [Cofactor](https://mathworld.wolfram.com/Cofactor.html "Cofactor")
*   [Adjugate matrix](https://en.wikipedia.org/wiki/Adjugate_matrix "Adjugate matrix")
*   [Singular Matrix](https://mathworld.wolfram.com/SingularMatrix.html "Singular Matrix")
*   [Elementary Matrix Operations](https://stattrek.com/matrix-algebra/elementary-operations "Elementary Matrix Operations")
*   [Gaussian Elimination](https://mathworld.wolfram.com/GaussianElimination.html "Gaussian Elimination")
*   [Gauss-Jordan Elimination](https://mathworld.wolfram.com/Gauss-JordanElimination.html "Gauss-Jordan Elimination")
*   [Matrix Inverse](https://mathworld.wolfram.com/MatrixInverse.html "Matrix Inverse")
*   [Eigenvectors and eigenvalues | Essence of linear algebra](https://www.youtube.com/watch?v=PFDu9oVAE-g "Eigenvectors and eigenvalues | Essence of linear algebra")
*   [Eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors "Eigenvalues and eigenvectors")
*   [Eigenvalues and Eigenvectors](https://math.mit.edu/~gs/linearalgebra/ila6/ila6_6_1.pdf "Eigenvalues and Eigenvectors")
*   [Definiteness of a matrix](https://en.wikipedia.org/wiki/Definite_matrix "Definiteness of a matrix") **Up to Eigenvalues**
*   [Definite, Semi-Definite and Indefinite Matrices](http://mathonline.wikidot.com/definite-semi-definite-and-indefinite-matrices "Definite, Semi-Definite and Indefinite Matrices") **Ignore Hessian Matrices**
*   [Tests for Positive Definiteness of a Matrix](https://www.gaussianwaves.com/2013/04/tests-for-positive-definiteness-of-a-matrix/ "Tests for Positive Definiteness of a Matrix")
*   [Positive Definite Matrices and Minima](https://www.youtube.com/watch?v=tccVVUnLdbc "Positive Definite Matrices and Minima")
*   [Positive Definite Matrices](https://www.math.utah.edu/~zwick/Classes/Fall2012_2270/Lectures/Lecture33_with_Examples.pdf "Positive Definite Matrices")
* [numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)
## Learning Objectives
*   What is a determinant? How would you calculate it?
*   What is a minor, cofactor, adjugate? How would calculate them?
*   What is an inverse? How would you calculate it?
*   What are eigenvalues and eigenvectors? How would you calculate them?
*   What is definiteness of a matrix? How would you determine a matrix’s definiteness?

## Tasks

### 0\. Determinant

Write a function `def determinant(matrix):` that calculates the determinant of a matrix:

* `matrix` is a list of lists whose determinant should be calculated
* If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
* If `matrix` is not square, raise a `ValueError` with the message `matrix must be a square matrix`
* The list `[[]]` represents a `0x0` matrix
* Returns: the determinant of `matrix`


`alexa@ubuntu-xenial:advanced_linear_algebra$ cat 0-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
 determinant = __import__('0-determinant').determinant

 mat0 = [[]]
 mat1 = [[5]]
 mat2 = [[1, 2], [3, 4]]
 mat3 = [[1, 1], [1, 1]]
 mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
 mat5 = []
 mat6 = [[1, 2, 3], [4, 5, 6]]

 print(determinant(mat0))
 print(determinant(mat1))
 print(determinant(mat2))
 print(determinant(mat3))
 print(determinant(mat4))
 try:
 determinant(mat5)
 except Exception as e:
 print(e)
 try:
 determinant(mat6)
 except Exception as e:
 print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./0-main.py 
1
5
-2
0
192
matrix must be a list of lists
matrix must be a square matrix
alexa@ubuntu-xenial:advanced_linear_algebra$`
### 1\. Minor

Write a function `def minor(matrix):` that calculates the minor matrix of a matrix:

* `matrix` is a list of lists whose minor matrix should be calculated
* If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
* If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
* Returns: the minor matrix of `matrix`


`alexa@ubuntu-xenial:advanced_linear_algebra$ cat 1-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
 minor = __import__('1-minor').minor

 mat1 = [[5]]
 mat2 = [[1, 2], [3, 4]]
 mat3 = [[1, 1], [1, 1]]
 mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
 mat5 = []
 mat6 = [[1, 2, 3], [4, 5, 6]]

 print(minor(mat1))
 print(minor(mat2))
 print(minor(mat3))
 print(minor(mat4))
 try:
 minor(mat5)
 except Exception as e:
 print(e)
 try:
 minor(mat6)
 except Exception as e:
 print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./1-main.py 
[[1]]
[[4, 3], [2, 1]]
[[1, 1], [1, 1]]
[[-12, -36, 0], [10, -34, -32], [47, 13, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:advanced_linear_algebra$`
### 2\. Cofactor

Write a function `def cofactor(matrix):` that calculates the cofactor matrix of a matrix:

* `matrix` is a list of lists whose cofactor matrix should be calculated
* If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
* If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
* Returns: the cofactor matrix of `matrix`


`alexa@ubuntu-xenial:advanced_linear_algebra$ cat 2-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
 cofactor = __import__('2-cofactor').cofactor

 mat1 = [[5]]
 mat2 = [[1, 2], [3, 4]]
 mat3 = [[1, 1], [1, 1]]
 mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
 mat5 = []
 mat6 = [[1, 2, 3], [4, 5, 6]]

 print(cofactor(mat1))
 print(cofactor(mat2))
 print(cofactor(mat3))
 print(cofactor(mat4))
 try:
 cofactor(mat5)
 except Exception as e:
 print(e)
 try:
 cofactor(mat6)
 except Exception as e:
 print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./2-main.py 
[[1]]
[[4, -3], [-2, 1]]
[[1, -1], [-1, 1]]
[[-12, 36, 0], [-10, -34, 32], [47, -13, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:advanced_linear_algebra$`
### 3\. Adjugate

Write a function `def adjugate(matrix):` that calculates the adjugate matrix of a matrix:

* `matrix` is a list of lists whose adjugate matrix should be calculated
* If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
* If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
* Returns: the adjugate matrix of `matrix`


`alexa@ubuntu-xenial:advanced_linear_algebra$ cat 3-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
 adjugate = __import__('3-adjugate').adjugate

 mat1 = [[5]]
 mat2 = [[1, 2], [3, 4]]
 mat3 = [[1, 1], [1, 1]]
 mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
 mat5 = []
 mat6 = [[1, 2, 3], [4, 5, 6]]

 print(adjugate(mat1))
 print(adjugate(mat2))
 print(adjugate(mat3))
 print(adjugate(mat4))
 try:
 adjugate(mat5)
 except Exception as e:
 print(e)
 try:
 adjugate(mat6)
 except Exception as e:
 print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./3-main.py 
[[1]]
[[4, -2], [-3, 1]]
[[1, -1], [-1, 1]]
[[-12, -10, 47], [36, -34, -13], [0, 32, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:advanced_linear_algebra$`
### 4\. Inverse

Write a function `def inverse(matrix):` that calculates the inverse of a matrix:

* `matrix` is a list of lists whose inverse should be calculated
* If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
* If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
* Returns: the inverse of `matrix`, or `None` if `matrix` is singular


`alexa@ubuntu-xenial:advanced_linear_algebra$ cat 4-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
 inverse = __import__('4-inverse').inverse

 mat1 = [[5]]
 mat2 = [[1, 2], [3, 4]]
 mat3 = [[1, 1], [1, 1]]
 mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
 mat5 = []
 mat6 = [[1, 2, 3], [4, 5, 6]]

 print(inverse(mat1))
 print(inverse(mat2))
 print(inverse(mat3))
 print(inverse(mat4))
 try:
 inverse(mat5)
 except Exception as e:
 print(e)
 try:
 inverse(mat6)
 except Exception as e:
 print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./4-main.py 
[[0.2]]
[[-2.0, 1.0], [1.5, -0.5]]
None
[[-0.0625, -0.052083333333333336, 0.24479166666666666], [0.1875, -0.17708333333333334, -0.06770833333333333], [0.0, 0.16666666666666666, -0.08333333333333333]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:advanced_linear_algebra$`
### 5\. Definiteness

Write a function `def definiteness(matrix):` that calculates the definiteness of a matrix:

* `matrix` is a `numpy.ndarray` of shape `(n, n)` whose definiteness should be calculated
* If `matrix` is not a `numpy.ndarray`, raise a `TypeError` with the message `matrix must be a numpy.ndarray`
* If `matrix` is not a valid matrix, return `None`
* Return: the string `Positive definite`, `Positive semi-definite`, `Negative semi-definite`, `Negative definite`, or `Indefinite` if the matrix is positive definite, positive semi\-definite, negative semi\-definite, negative definite of indefinite, respectively
* If `matrix` does not fit any of the above categories, return `None`
* You may `import numpy as np`


`alexa@ubuntu-xenial:advanced_linear_algebra$ cat 5-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
 definiteness = __import__('5-definiteness').definiteness
 import numpy as np

 mat1 = np.array([[5, 1], [1, 1]])
 mat2 = np.array([[2, 4], [4, 8]])
 mat3 = np.array([[-1, 1], [1, -1]])
 mat4 = np.array([[-2, 4], [4, -9]])
 mat5 = np.array([[1, 2], [2, 1]])
 mat6 = np.array([])
 mat7 = np.array([[1, 2, 3], [4, 5, 6]])
 mat8 = [[1, 2], [1, 2]]

 print(definiteness(mat1))
 print(definiteness(mat2))
 print(definiteness(mat3))
 print(definiteness(mat4))
 print(definiteness(mat5))
 print(definiteness(mat6))
 print(definiteness(mat7))
 try:
 definiteness(mat8)
 except Exception as e:
 print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./5-main.py 
Positive definite
Positive semi-definite
Negative semi-definite
Negative definite
Indefinite
None
None
matrix must be a numpy.ndarray
alexa@ubuntu-xenial:advanced_linear_algebra$`
