import numpy as np

class MathOperations:

    def gradientX(p_matrix, delta_x):
        p_matrix_no_first_col = p_matrix[:,1:]
        p_matrix_no_last_col = p_matrix[:,:-1]

        results = p_matrix_no_first_col - p_matrix_no_last_col
        results = results/delta_x
        return results

    def gradientY(p_matrix, delta_y):
        p_matrix_no_first_col = p_matrix[1:,:]
        p_matrix_no_last_col = p_matrix[:-1,:]

        results = p_matrix_no_first_col - p_matrix_no_last_col
        results = (results.transpose()/delta_y).transpose()
        return results

    def divergence(velocity_x, velocity_y):
        velocity_x_no_first_col = velocity_x[:,1:]
        velocity_x_no_last_col = velocity_x[:,:-1]
        p_matrix_no_first_col = p_matrix[1:,:]
        p_matrix_no_last_col = p_matrix[:-1,:]