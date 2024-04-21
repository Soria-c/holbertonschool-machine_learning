#!/usr/bin/env python3
"""Matplotlib tasks"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Function to plot a stacked bar chart"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    persons = ["Farrah", "Fred", "Felicia"]
    colors = ("r", "yellow", "#ff8000", "#ffe5b4")
    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")
    plt.gca().set_ylim([0, 80])
    plt.yticks(np.linspace(0, 80, 9))
    bottom = np.zeros(3)
    for index, row in enumerate(fruit):
        plt.bar(persons, row, color=colors[index], width=0.5, bottom=bottom)
        bottom += row
    plt.legend(("apples", "bananas", "oranges", "peaches"))
    plt.show()
