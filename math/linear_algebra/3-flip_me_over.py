#!/usr/bin/env python3
def matrix_transpose(matrix):
    new_martix = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
            new_martix.append(row)
    return new_martix
