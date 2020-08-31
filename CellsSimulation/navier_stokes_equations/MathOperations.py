import numpy as np


class MathOperations:

    @staticmethod
    def gradientX(p_matrix, delta_x):
        p_matrix_no_first_col = p_matrix[:, 1:]
        p_matrix_no_last_col = p_matrix[:, :-1]

        results = p_matrix_no_first_col - p_matrix_no_last_col
        results = results / delta_x
        return results

    @staticmethod
    def gradientY(p_matrix, delta_y):
        p_matrix_no_first_col = p_matrix[1:, :]
        p_matrix_no_last_col = p_matrix[:-1, :]

        results = p_matrix_no_first_col - p_matrix_no_last_col
        results = (results.transpose() / delta_y).transpose()
        return results

    @staticmethod
    def divergence(velocity_x, velocity_y):
        # P = plus, M = minus
        u_i_j = velocity_x[1:, 1:]
        u_iM1_j = velocity_x[:-1, 1:]
        v_i_j = velocity_y[1:, 1:]
        v_i_jM1 = velocity_y[1:, :-1]
        return u_iM1_j + u_iM1_j + v_i_j + v_i_jM1
