# Linear Algebra
## Read or watch
* [Introduction to vectors](https://www.youtube.com/watch?v=fNk_zzaMoSs)
* [What is a matrix? (not the matrix)](https://math.stackexchange.com/questions/2782717/what-exactly-is-a-matrix)
* [Transpose](https://en.wikipedia.org/wiki/Transpose)
* [Understanding the dot product](https://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/)
* [Matrix Multiplication](https://www.youtube.com/watch?v=BzWahqwaS8k)
* [What is the relationship between matrix multiplication and the dot product?](https://www.quora.com/What-is-the-relationship-between-matrix-multiplication-and-the-dot-product)
* [The Dot Product, Matrix Multiplication, and the Magic of Orthogonal Matrices (advanced)](https://www.youtube.com/watch?v=rW2ypKLLxGk)
* [numpy tutorial (until Shape Manipulation (excluded))](https://numpy.org/doc/stable/user/quickstart.html)
* [numpy basics (until Universal Functions (included))](https://www.oreilly.com/library/view/python-for-data/9781449323592/ch04.html)
* [array indexing](https://docs.scipy.org/doc/numpy-1.15.0/reference/arrays.indexing.html#basic-slicing-and-indexing)
* [numerical operations on arrays](https://scipy-lectures.org/intro/numpy/operations.html)
* [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
* [numpy mutations and broadcasting](https://towardsdatascience.com/two-cool-features-of-python-numpy-mutating-by-slicing-and-broadcasting-3b0b86e8b4c7)
* [numpy.ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html)
* [numpy.ndarray.shape](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html)
* [numpy.transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
* [numpy.ndarray.transpose](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html)
* [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)

## Learning Objectives
* What is a vector?
* What is a matrix?
* What is a transpose?
* What is the shape of a matrix?
* What is an axis?
* What is a slice?
* How do you slice a vector/matrix?
* What are element-wise operations?
* How do you concatenate vectors/matrices?
* What is the dot product?
* What is matrix multiplication?
* What is Numpy?
* What is parallelization and why is it important?
* What is broadcasting?

## Tasks
### [0. Slice Me Up](./0-main.ipynb)
Complete the following source code:
```python
#!/usr/bin/env python3
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]
arr1 =  # your code here
arr2 =  # your code here
arr3 =  # your code here
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
```
* **arr1** should be the first two numbers of **arr**
* **arr2** should be the last five numbers of **arr**
* **arr3** should be the 2nd through 6th numbers of **arr**
* You are not allowed to use any loops or conditional statements
* Your program should be exactly 8 lines
```python
The first two numbers of the array are: [9, 8]
The last five numbers of the array are: [9, 4, 1, 0, 3]
The 2nd through 6th numbers of the array are: [8, 2, 3, 9, 4]
```
### [1. Trim Me Down](./1-main.ipynb)
Complete the following source code:
```python
#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
# your code here
print("The middle columns of the matrix are: {}".format(the_middle))
```
* **the_middle** should be a 2D matrix containing the 3rd and  4th columns of **matrix**
* You are not allowed to use any conditional statements
* You are only allowed to use one for loop
* Your program should be exactly 6 lines
```python
The middle columns of the matrix are: [[9, 4], [7, 3], [4, 6]]
```
### [2. Size Me Please](./2-main.ipynb)
Write a function **def matrix_shape(matrix):** that calculates the shape of a matrix:

* You can assume all elements in the same dimension are of the same type/shape
* The shape should be returned as a list of integers
```python
matrix_shape = __import__('2-size_me_please').matrix_shape

mat1 = [[1, 2], [3, 4]]
print(matrix_shape(mat1))
mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
print(matrix_shape(mat2))

```
```python
[2, 2]
[2, 3, 5]
```
### [3. Flip Me Over](./3-main.ipynb)
Write a function **def matrix_transpose(matrix):** that returns the transpose of a 2D matrix, **matrix**:
* You must return a new matrix
* You can assume that **matrix** is never empty
* You can assume all elements in the same dimension are of the same type/shape
```python
matrix_transpose = __import__('3-flip_me_over').matrix_transpose

mat1 = [[1, 2], [3, 4]]
print(mat1)
print(matrix_transpose(mat1))
mat2 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
print(mat2)
print(matrix_transpose(mat2))
```
```python
[[1, 2], [3, 4]]
[[1, 3], [2, 4]]

[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
[[1, 6, 11, 16, 21, 26], [2, 7, 12, 17, 22, 27], [3, 8, 13, 18, 23, 28], [4, 9, 14, 19, 24, 29], [5, 10, 15, 20, 25, 30]]
```
### [4. Line Up](./4-main.ipynb)
Write a function **def add_arrays(arr1, arr2):** that adds two arrays element-wise:
* You can assume that **arr1** and **arr2** are lists of ints/floats
* You must return a new list
* If **arr1** and **arr2** are not the same shape, return **None**
```python
#!/usr/bin/env python3

add_arrays = __import__('4-line_up').add_arrays

arr1 = [1, 2, 3, 4]
arr2 = [5, 6, 7, 8]
print(add_arrays(arr1, arr2))
print(arr1)
print(arr2)
print(add_arrays(arr1, [1, 2, 3]))
```
```python
[6, 8, 10, 12]
[1, 2, 3, 4]
[5, 6, 7, 8]
None
```
### [5. Across The Planes](./5-main.ipynb)
Write a function **def add_matrices2D(mat1, mat2):** that adds two matrices element-wise:
* You can assume that **mat1** and **mat2** are 2D matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If **mat1** and **mat2** are not the same shape, return **None**
```python
#!/usr/bin/env python3

add_matrices2D = __import__('5-across_the_planes').add_matrices2D

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6], [7, 8]]
print(add_matrices2D(mat1, mat2))
print(mat1)
print(mat2)
print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
```
```python
[[6, 8], [10, 12]]
[[1, 2], [3, 4]]
[[5, 6], [7, 8]]
None
```
### [6. Howdy Partner](./6-main.ipynb)
Write a function **def cat_arrays(arr1, arr2):** that concatenates two arrays:
* You can assume that **arr1** and **arr2** are lists of ints/floats
* You must return a new list
```python
#!/usr/bin/env python3

cat_arrays = __import__('6-howdy_partner').cat_arrays

arr1 = [1, 2, 3, 4, 5]
arr2 = [6, 7, 8]
print(cat_arrays(arr1, arr2))
print(arr1)
print(arr2)
```
```python
[1, 2, 3, 4, 5, 6, 7, 8]
[1, 2, 3, 4, 5]
[6, 7, 8]
```
### [7. Gettin’ Cozy](./7-main.ipynb)
Write a function **def cat_matrices2D(mat1, mat2, axis=0):** that concatenates two matrices along a specific axis:
* You can assume that **mat1** and **mat2** are 2D matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If the two matrices cannot be concatenated, return **None**
```python
#!/usr/bin/env python3

cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)
mat5 = cat_matrices2D(mat1, mat3, axis=1)
print(mat4)
print(mat5)
mat1[0] = [9, 10]
mat1[1].append(5)
print(mat1)
print(mat4)
print(mat5)
```
```python
[[1, 2], [3, 4], [5, 6]]
[[1, 2, 7], [3, 4, 8]]
[[9, 10], [3, 4, 5]]
[[1, 2], [3, 4], [5, 6]]
[[1, 2, 7], [3, 4, 8]]
```
### [8. Ridin’ Bareback](./8-main.ipynb)
Write a function **def mat_mul(mat1, mat2):** that performs matrix multiplication:
* You can assume that **mat1** and **mat2** are 2D matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If the two matrices cannot be multiplied, return **None**
```python
#!/usr/bin/env python3

mat_mul = __import__('8-ridin_bareback').mat_mul

mat1 = [[1, 2],
        [3, 4],
        [5, 6]]
mat2 = [[1, 2, 3, 4],
        [5, 6, 7, 8]]
print(mat_mul(mat1, mat2))
```
```python
[[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]
```
### [9. Let The Butcher Slice It](./9-main.ipynb)
Complete the following source code (found below):
* **mat1** should be the middle two rows of **matrix**
* **mat2** should be the middle two columns of **matrix**
* **mat3** should be the bottom-right, square, 3x3 matrix of **matrix**
* You are not allowed to use any loops or conditional statements
* Your program should be exactly 10 lines
```python
#!/usr/bin/env python3
import numpy as np
matrix = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]])
mat1 =  # your code here
mat2 =  # your code here
mat3 =  # your code here
print("The middle two rows of the matrix are:\n{}".format(mat1))
print("The middle two columns of the matrix are:\n{}".format(mat2))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))
```
```python
The middle two rows of the matrix are:
[[ 7  8  9 10 11 12]
 [13 14 15 16 17 18]]
The middle two columns of the matrix are:
[[ 3  4]
 [ 9 10]
 [15 16]
 [21 22]]
The bottom-right, square, 3x3 matrix is:
[[10 11 12]
 [16 17 18]
 [22 23 24]]
```
### [10. I’ll Use My Scale](./10-main.ipynb)
Write a function **def np_shape(matrix):** that calculates the shape of a **numpy.ndarray**:
* You are not allowed to use any loops or conditional statements
* You are not allowed to use **try/except** statements
* The shape should be returned as a tuple of integers
```python
#!/usr/bin/env python3

import numpy as np
np_shape = __import__('10-ill_use_my_scale').np_shape

mat1 = np.array([1, 2, 3, 4, 5, 6])
mat2 = np.array([])
mat3 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                 [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
print(np_shape(mat1))
print(np_shape(mat2))
print(np_shape(mat3))
```
```python
(6,)
(0,)
(2, 2, 5)
```
### [11. The Western Exchange](./11-main.ipynb)
Write a function **def np_transpose(matrix):** that transposes **matrix**:
* You can assume that **matrix** can be interpreted as a **numpy.ndarray**
* You are not allowed to use any loops or conditional statements
* You must return a new **numpy.ndarray**
```python
#!/usr/bin/env python3

import numpy as np
np_transpose = __import__('11-the_western_exchange').np_transpose

mat1 = np.array([1, 2, 3, 4, 5, 6])
mat2 = np.array([])
mat3 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                 [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
print(np_transpose(mat1))
print(mat1)
print(np_transpose(mat2))
print(mat2)
print(np_transpose(mat3))
print(mat3)
```
```python
[1 2 3 4 5 6]
[1 2 3 4 5 6]
[]
[]
[[[ 1 11]
  [ 6 16]]

 [[ 2 12]
  [ 7 17]]

 [[ 3 13]
  [ 8 18]]

 [[ 4 14]
  [ 9 19]]

 [[ 5 15]
  [10 20]]]
[[[ 1  2  3  4  5]
  [ 6  7  8  9 10]]

 [[11 12 13 14 15]
  [16 17 18 19 20]]]
```
### [12. Bracing The Elements](./12-main.ipynb)
Write a function **def np_elementwise(mat1, mat2):** that performs element-wise addition, subtraction, multiplication, and division:
* You can assume that **mat1** and **mat2** can be interpreted as numpy.ndarrays
* You should return a tuple containing the element-wise sum, difference, product, and quotient, respectively
* You are not allowed to use any loops or conditional statements
* You can assume that **mat1** and **mat2** are never empty
```python
#!/usr/bin/env python3

import numpy as np
np_elementwise = __import__('12-bracin_the_elements').np_elementwise

mat1 = np.array([[11, 22, 33], [44, 55, 66]])
mat2 = np.array([[1, 2, 3], [4, 5, 6]])

print(mat1)
print(mat2)
add, sub, mul, div = np_elementwise(mat1, mat2)
print("Add:\n", add, "\nSub:\n", sub, "\nMul:\n", mul, "\nDiv:\n", div)
add, sub, mul, div = np_elementwise(mat1, 2)
print("Add:\n", add, "\nSub:\n", sub, "\nMul:\n", mul, "\nDiv:\n", div)
```
```python
[[11 22 33]
 [44 55 66]]
[[1 2 3]
 [4 5 6]]
Add:
 [[12 24 36]
 [48 60 72]] 
Sub:
 [[10 20 30]
 [40 50 60]] 
Mul:
 [[ 11  44  99]
 [176 275 396]] 
Div:
 [[11. 11. 11.]
 [11. 11. 11.]]
Add:
 [[13 24 35]
 [46 57 68]] 
Sub:
 [[ 9 20 31]
 [42 53 64]] 
Mul:
 [[ 22  44  66]
 [ 88 110 132]] 
Div:
 [[ 5.5 11.  16.5]
 [22.  27.5 33. ]]
```
### [13. Cat's Got Your Tongue](./13-main.ipynb)
Write a function **def np_cat(mat1, mat2, axis=0):** that concatenates two matrices along a specific axis:
* You can assume that **mat1** and **mat2** can be interpreted as numpy.ndarrays
* You must return a new **numpy.ndarray**
* You are not allowed to use any loops or conditional statements
* You may use: **import numpy as np**
* You can assume that **mat1** and **mat2** are never empty
```python
#!/usr/bin/env python3

import numpy as np
np_cat = __import__('13-cats_got_your_tongue').np_cat

mat1 = np.array([[11, 22, 33], [44, 55, 66]])
mat2 = np.array([[1, 2, 3], [4, 5, 6]])
mat3 = np.array([[7], [8]])
print(np_cat(mat1, mat2))
print(np_cat(mat1, mat2, axis=1))
print(np_cat(mat1, mat3, axis=1))
```
```python
[[11 22 33]
 [44 55 66]
 [ 1  2  3]
 [ 4  5  6]]
[[11 22 33  1  2  3]
 [44 55 66  4  5  6]]
[[11 22 33  7]
 [44 55 66  8]]
```
### [14. Saddle Up](./14-main.ipynb)
Write a function **def np_matmul(mat1, mat2):** that performs matrix multiplication:
* You can assume that **mat1** and **mat2** are numpy.ndarrays
* You are not allowed to use any loops or conditional statements
* You may use: **import numpy as np**
* You can assume that **mat1** and **mat2** are never empty
```python
#!/usr/bin/env python3

import numpy as np
np_matmul = __import__('14-saddle_up').np_matmul

mat1 = np.array([[11, 22, 33], [44, 55, 66]])
mat2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mat3 = np.array([[7], [8], [9]])
print(np_matmul(mat1, mat2))
print(np_matmul(mat1, mat3))
```
```python
[[ 330  396  462]
 [ 726  891 1056]]
[[ 550]
 [1342]]
```
### [15. Slice Like A Ninja](./100-main%20.ipynb)
Write a function **def np_slice(matrix, axes={}):** that slices a matrix along specific axes:
* You can assume that matrix is a numpy.ndarray
* You must return a new numpy.ndarray
axes is a dictionary where the key is an axis to slice along and the value is a tuple representing the slice to make along that axis
* You can assume that axes represents a valid slice
* [Hint](https://docs.python.org/3/library/functions.html#slice)
```python
#!/usr/bin/env python3

import numpy as np
np_slice = __import__('100-slice_like_a_ninja').np_slice

mat1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(np_slice(mat1, axes={1: (1, 3)}))
print(mat1)
mat2 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                 [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                 [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
print(mat2)
```
```python
[[2 3]
 [7 8]]
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
[[[ 5  3  1]
  [10  8  6]]

 [[15 13 11]
  [20 18 16]]]
[[[ 1  2  3  4  5]
  [ 6  7  8  9 10]]

 [[11 12 13 14 15]
  [16 17 18 19 20]]

 [[21 22 23 24 25]
  [26 27 28 29 30]]]
```
### [16. The Whole Barn](./101-main.ipynb)
Write a function **def add_matrices(mat1, mat2):** that adds two matrices:
* You can assume that **mat1** and **mat2** are matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If matrices are not the same shape, return **None**
* You can assume that **mat1** and **mat2** will never be empty
```python
#!/usr/bin/env python3

add_matrices = __import__('101-the_whole_barn').add_matrices

mat1 = [1, 2, 3]
mat2 = [4, 5, 6]
print(add_matrices(mat1, mat2))
mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6], [7, 8]]
print(add_matrices(mat1, mat2))
mat1 = [[[[1, 2, 3, 4], [5, 6, 7, 8]],
         [[9, 10, 11, 12], [13, 14 ,15, 16]],
         [[17, 18, 19, 20], [21, 22, 23, 24]]],
        [[[25, 26, 27, 28], [29, 30, 31, 32]],
         [[33, 34, 35, 36], [37, 38, 39, 40]],
         [[41, 42, 43, 44], [45, 46, 47, 48]]]]
mat2 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
         [[19, 110, 111, 112], [113, 114 ,115, 116]],
         [[117, 118, 119, 120], [121, 122, 123, 124]]],
        [[[125, 126, 127, 128], [129, 130, 131, 132]],
         [[133, 134, 135, 136], [137, 138, 139, 140]],
         [[141, 142, 143, 144], [145, 146, 147, 148]]]]
mat3 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
         [[117, 118, 119, 120], [121, 122, 123, 124]]],
        [[[125, 126, 127, 128], [129, 130, 131, 132]],
         [[141, 142, 143, 144], [145, 146, 147, 148]]]]
print(add_matrices(mat1, mat2))
print(add_matrices(mat1, mat3))
```
```python
[5, 7, 9]
[[6, 8], [10, 12]]
[[[[12, 14, 16, 18], [20, 22, 24, 26]], [[28, 120, 122, 124], [126, 128, 130, 132]], [[134, 136, 138, 140], [142, 144, 146, 148]]], [[[150, 152, 154, 156], [158, 160, 162, 164]], [[166, 168, 170, 172], [174, 176, 178, 180]], [[182, 184, 186, 188], [190, 192, 194, 196]]]]
None
```
### [17. Squashed Like Sardines](./102-main.ipynb)
Write a function **def cat_matrices(mat1, mat2, axis=0):** that concatenates two matrices along a specific axis:
* You can assume that **mat1** and **mat2** are matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If you cannot concatenate the matrices, return **None**
* You can assume that **mat1** and **mat2** are never empty
- *Note the time difference between the standard Python3 library and the numpy library is an order of magnitude! When you have matrices with millions of data points, this time adds up!*
```python
#!/usr/bin/env python3

import numpy as np
import time
cat_matrices = __import__('102-squashed_like_sardines').cat_matrices

mat1 = [1, 2, 3]
mat2 = [4, 5, 6]
np_mat1 = np.array(mat1)
np_mat2 = np.array(mat2)

t0 = time.time()
m = cat_matrices(mat1, mat2)
t1 = time.time()
print(t1 - t0)
print(m)
t0 = time.time()
np.concatenate((np_mat1, np_mat2))
t1 = time.time()
print(t1 - t0, "\n")

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6], [7, 8]]
np_mat1 = np.array(mat1)
np_mat2 = np.array(mat2)

t0 = time.time()
m = cat_matrices(mat1, mat2)
t1 = time.time()
print(t1 - t0)
print(m)
t0 = time.time()
np.concatenate((np_mat1, np_mat2))
t1 = time.time()
print(t1 - t0, "\n")

t0 = time.time()
m = cat_matrices(mat1, mat2, axis=1)
t1 = time.time()
print(t1 - t0)
print(m)
t0 = time.time()
np.concatenate((mat1, mat2), axis=1)
t1 = time.time()
print(t1 - t0, "\n")

mat3 = [[[[1, 2, 3, 4], [5, 6, 7, 8]],
         [[9, 10, 11, 12], [13, 14 ,15, 16]],
         [[17, 18, 19, 20], [21, 22, 23, 24]]],
        [[[25, 26, 27, 28], [29, 30, 31, 32]],
         [[33, 34, 35, 36], [37, 38, 39, 40]],
         [[41, 42, 43, 44], [45, 46, 47, 48]]]]
mat4 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
         [[19, 110, 111, 112], [113, 114 ,115, 116]],
         [[117, 118, 119, 120], [121, 122, 123, 124]]],
        [[[125, 126, 127, 128], [129, 130, 131, 132]],
         [[133, 134, 135, 136], [137, 138, 139, 140]],
         [[141, 142, 143, 144], [145, 146, 147, 148]]]]
mat5 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
         [[117, 118, 119, 120], [121, 122, 123, 124]]],
        [[[125, 126, 127, 128], [129, 130, 131, 132]],
         [[141, 142, 143, 144], [145, 146, 147, 148]]]]
np_mat3 = np.array(mat3)
np_mat4 = np.array(mat4)
np_mat5 = np.array(mat5)

t0 = time.time()
m = cat_matrices(mat3, mat4, axis=3)
t1 = time.time()
print(t1 - t0)
print(m)
t0 = time.time()
np.concatenate((np_mat3, np_mat4), axis=3)
t1 = time.time()
print(t1 - t0, "\n")

t0 = time.time()
m = cat_matrices(mat3, mat5, axis=1)
t1 = time.time()
print(t1 - t0)
print(m)
t0 = time.time()
np.concatenate((np_mat3, np_mat5), axis=1)
t1 = time.time()
print(t1 - t0, "\n")

m = cat_matrices(mat2, mat5)
print(m)
```
```python
1.6927719116210938e-05
[1, 2, 3, 4, 5, 6]
4.76837158203125e-06 

1.8358230590820312e-05
[[1, 2], [3, 4], [5, 6], [7, 8]]
3.0994415283203125e-06 

1.7881393432617188e-05
[[1, 2, 5, 6], [3, 4, 7, 8]]
6.9141387939453125e-06 

0.00016427040100097656
[[[[1, 2, 3, 4, 11, 12, 13, 14], [5, 6, 7, 8, 15, 16, 17, 18]], [[9, 10, 11, 12, 19, 110, 111, 112], [13, 14, 15, 16, 113, 114, 115, 116]], [[17, 18, 19, 20, 117, 118, 119, 120], [21, 22, 23, 24, 121, 122, 123, 124]]], [[[25, 26, 27, 28, 125, 126, 127, 128], [29, 30, 31, 32, 129, 130, 131, 132]], [[33, 34, 35, 36, 133, 134, 135, 136], [37, 38, 39, 40, 137, 138, 139, 140]], [[41, 42, 43, 44, 141, 142, 143, 144], [45, 46, 47, 48, 145, 146, 147, 148]]]]
5.030632019042969e-05 

0.00020313262939453125
[[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]], [[11, 12, 13, 14], [15, 16, 17, 18]], [[117, 118, 119, 120], [121, 122, 123, 124]]], [[[25, 26, 27, 28], [29, 30, 31, 32]], [[33, 34, 35, 36], [37, 38, 39, 40]], [[41, 42, 43, 44], [45, 46, 47, 48]], [[125, 126, 127, 128], [129, 130, 131, 132]], [[141, 142, 143, 144], [145, 146, 147, 148]]]]
1.5735626220703125e-05 

None
```