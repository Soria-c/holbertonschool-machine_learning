# Decision_Tree.pred vs Decision_Tree.predict 
* Predicting the class of an individual in a decision tree can be done by the following simple procedure:
```python
# in the Leaf class, add :
    def pred(self,x) :
        return self.value

# in the Node class, add :
    def pred(self,x) :
        if x[self.feature]>self.threshold :
            return self.left_child.pred(x)
        else :
            return self.right_child.pred(x)

# in the Decision_Tree class, add :
    def pred(self,x) :
            return self.root.pred(x)
```
* 

** For testing:** Insert the above methods in their respective classes.
* Main: For example, with these methods added in their respective classes, when evaluating
```python
print( example_0().pred(  np.array([1,22000])) ) 
print( example_0().pred(  np.array([0,22000])) )
print( example_1(4).pred( np.array([11.65]))   )
print( example_1(4).pred( np.array([6.917]))   )
```
* We obtain
```python
1
12
7
```
* Problem: Now if we have a large number of individuals whose explanatory features are stored in a numpy array `A` of shape `(n_individuals,n_features)` we can get the corresponding predictions by evaluating
* Problem: Now if we have a large number of individuals whose explanatory features are stored in a numpy array `A` of shape `(n_individuals,n_features)` we can get the corresponding predictions by evaluating
* `[T.pred(x) for x in A]`
* Solution: The forthcoming tasks are designed to get a faster prediction tool in the form of an attribute `Decision_Tree.predict` of our tree.
* We will apply the following strategy :
    * First we will compute the bounds of each leaf. Indeed there is a set of inequalities that completely define whether an individual crosses a given `node` or not. To be explicit, these inequalities are of the form `A[:,anode.feature]>anode.threshold` or `A[:,anode.feature]<=anode.threshold` for each ancestor `anode` of `node`
    * Once these bounds are known (and recorded) for each leaf, we can compute the indicator function of a given leaf : a function that associates to A a 1D numpy array of booleans of size `n_individuals` (the i-th element of the later is `True` iff the i-th individual falls in the leaf).
    * Once these indicator functions are known (and recorded) for each leaf, we compute the 1D array of predictions as the sum of the products `leaf.indicator(A)*leaf.value`
* Example: Once the `Decision_Tree.predict` attribute will be computed, we will be able to compare its efficiency with that of the above defined `Decision_Tree.pred`. Here is a report of such a comparison ran on an i5-5257U CPU with a 16GB RAM .
```python
import time
def time_function (f, n_iter=100) :
    def time_one_iter() :
        t=time.process_time()
        f()
        return time.process_time()-t
    times =np.array([time_one_iter() for _ in range(n_iter)])
    return  f"On {n_iter} iterations : , mean time = {times.mean()}, std = {times.std()}"


T=example_1(5)
T.update_predict()
N=100000

def with_pred() :
    explanatory = np.array(np.random.rand(N)*32).reshape([N,1])
    return np.array([T.pred(x) for x in explanatory])

def with_predict() :
    explanatory = np.array(np.random.rand(N)*32).reshape([N,1])
    return T.predict(explanatory)

print(f"With pred    : {time_function(with_pred)}")
print(f"With predict : {time_function(with_predict)}")
```
```python
With pred    : On 100 iterations : , mean time = 0.26521891637999956, std = 0.005345708513575958
With predict : On 100 iterations : , mean time = 0.01726781183000042, std = 0.0006482395464351253
```
* The superiority of `T.predict` over the more naive `T.pred` should now clearly appear.