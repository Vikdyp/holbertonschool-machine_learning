#!/usr/bin/env python3
def matrix_transpose(matrix):
    new_matrix = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        new_matrix.append(row)
    return new_matrix
