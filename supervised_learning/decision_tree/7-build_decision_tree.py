#!/usr/bin/env python3
"""This module defines several clases to implement a decision tree"""

import numpy as np


class Node:
    """Base node class"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Constructor"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Function to calculate the maximum depth"""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Function to calculate the number of internal nodes or/and leaves"""
        return int(not only_leaves) +\
            self.left_child.count_nodes_below(only_leaves) +\
            self.right_child.count_nodes_below(only_leaves)

    def right_child_add_prefix(self, text):
        """Function to format right child"""
        lines = text.split("\n")
        new_text = "    +--"+lines[0]
        for x in lines[1:]:
            new_text += "\n       "+x
        return new_text

    def left_child_add_prefix(self, text):
        """Function to format left child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  "+x) + "\n"
        return (new_text)

    def __str__(self):
        """Function to format a node"""
        text = f"{'root' if self.is_root else '-> node'} \
[feature={self.feature}, threshold={self.threshold}]\n"
        return text + self.left_child_add_prefix(self.left_child.__str__()) \
            + self.right_child_add_prefix(self.right_child.__str__())

    def get_leaves_below(self):
        """Function to get a list of leaves below"""
        return [*self.left_child.get_leaves_below(),
                *self.right_child.get_leaves_below()]

    def update_bounds_below(self):
        """Function to update the bounds"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        self.left_child.upper = {**self.upper}
        self.left_child.lower = {**self.lower, self.feature: self.threshold}
        self.right_child.upper = {**self.upper, self.feature: self.threshold}
        self.right_child.lower = {**self.lower}

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Function to update the indicator function"""
        def is_large_enough(x):
            """Function to specify a condition"""
            return np.array([np.greater(x[:, key], self.lower[key])
                             for key in list(self.lower.keys())])
            # <- fill the gap : this function returns a 1D numpy array of size
            # n_individuals` so that the `i`-th element of the later is `True
            # if the `i`-th individual has all its features > the lower bounds

        def is_small_enough(x):
            """Function to specify a condition"""
            return np.array([np.less_equal(x[:, key], self.upper[key])
                             for key in list(self.upper.keys())])
            # <- fill the gap : this function returns a 1D numpy array of size
            # `n_individuals` so that the `i`-th element of the later is `True
            # if the `i`-th individual has all its features <= the lower bounds

        self.indicator = lambda x: np.all(np.array(
            [*is_large_enough(x), *is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Function to predict a value"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Leaf class"""
    def __init__(self, value, depth=None):
        """Constructor"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Function to calculate the maximum depth"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Function to calculate the number of internal nodes or/and leaves"""
        return 1

    def __str__(self):
        """Function to format a leaf"""
        return (f"-> leaf [value={self.value}] ")

    def get_leaves_below(self):
        """Function to get a list of leaves below"""
        return [self]

    def update_bounds_below(self):
        """Function to update the bounds"""
        pass

    def pred(self, x):
        """Function to predict a value"""
        return self.value


class Decision_Tree():
    """Decision tree class"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Constructor"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Function to calculate the maximum depth"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Function to calculate the number of internal nodes or/and leaves"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Function to format a tree"""
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """Function to get a list of leaves below"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Function to update the bounds"""
        self.root.update_bounds_below()

    def pred(self, x):
        """Function to compute the predictions"""
        return self.root.pred(x)

    def update_predict(self):
        """Function to update the prediction"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum([leaf.indicator(A) * leaf.value
                                         for leaf in leaves], axis=0)

    def fit(self, explanatory, target, verbose=0):
        """Function to fir the tree"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }
    - Accuracy on training data : { self.accuracy(self.explanatory,
                                              self.target)}""")

    def np_extrema(self, arr):
        """"Function to find the extrema"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """"Splitting function"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self\
                .np_extrema(self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1-x) * feature_min + x*feature_max
        return feature, threshold

    def fit_node(self, node):
        """"Function to fit a node an their children"""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population &\
            (self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population &\
            (self.explanatory[:, node.feature] <= node.threshold)

        # Is left node a leaf ?

        left_filter = self.explanatory[:, node.feature][left_population]
        is_left_leaf = ((left_filter.size < self.min_pop)
                        or (node.depth + 1 == self.max_depth)
                        or (np.unique(self.target[left_population]).size
                            == 1))

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        right_filter = self.explanatory[:, node.feature][right_population]
        is_right_leaf = ((right_filter.size < self.min_pop)
                         or (node.depth + 1 == self.max_depth)
                         or (np.unique(self.target[right_population]).size
                             == 1))

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """"Function to get create a leaf node"""
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """"Function to get create a child node"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """"Function to compute accuracy"""
        return np.sum(np.equal(self.predict(test_explanatory), test_target))\
            / test_target.size
