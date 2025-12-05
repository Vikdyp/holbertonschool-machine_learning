#!/usr/bin/env python3
"""module that provide the frequency function"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """function that print an histogram"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.hist(student_grades, bins=10, edgecolor='black')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.show()
