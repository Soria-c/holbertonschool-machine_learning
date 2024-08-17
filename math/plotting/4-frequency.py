#!/usr/bin/env python3
"""Matplotlib tasks"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Function to plot a histogram"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    
    plt.title("Project A")
    plt.ylabel("Number of Students")
    plt.xlabel("Grades")
    bins = np.linspace(0, 100, 11)
    plt.hist(student_grades, bins=bins, edgecolor="black")
    plt.axis((0, 100, 0, 30))
    plt.xticks(bins)
    plt.show()
