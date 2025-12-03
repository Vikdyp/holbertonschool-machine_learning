#!/usr/bin/env python3
"""module qui contient la fonction matrix_transpose"""


def matrix_transpose(matrix):
    """fonction qui permet de retourner la transposer d'une matrice 2D"""
    new_matrix = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        new_matrix.append(row)
    return new_matrix
