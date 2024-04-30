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
