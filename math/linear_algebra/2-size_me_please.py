#!/usr/bin/env python3
"""module quuui contient la fonction pour calculer la forme d'une matrice"""


def matrix_shape(matrix):
    """calculate the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
