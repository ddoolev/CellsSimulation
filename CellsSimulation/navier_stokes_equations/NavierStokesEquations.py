import numpy as np
import scipy
from LaplaceOperator import LaplaceOperator


class NavierStokesEquations:
    def __init__(self, p_matrix, v_matrix, u_matrix, delta_x, delta_y):
        self.__p_matrix = p_matrix
        self.__v_matrix = v_matrix
        self.__u_matrix = u_matrix
        self.__delta_x = delta_x
        self.__delta_y = delta_y
        self.__laplacOperator = LaplaceOperator(delta_x, delta_y)

    # def getNextStep(self):
    def __non_linear_parameters_x(self):
        # P = plus  M = minus h = half
        # find the matrixes without the unneeded columns and rows
        u_iP1_j = self.__u_matrix[2:, 1:-1]
        u_i_j = self.__u_matrix[1:-1, 1:-1]
        u_iM1_j = self.__u_matrix[:-2, 1:-1]
        u_i_jP1 = self.__u_matrix[1:-1, 2:]
        u_i_jM1 = self.__u_matrix[1:-1, :-2]

        v_i_jP1 = self.__v_matrix[1:-1, 2:]
        v_iM1_jP1 = self.__v_matrix[:-2, 2:]
        v_i_j = self.__v_matrix[1:-1, 1:-1]
        v_iM1_j = self.__v_matrix[:-2, 1:-1]

        # find matrixes with half indexes
        u_iPh_j = (u_iP1_j + u_i_j) / 2
        u_i_jPh = (u_i_jP1 + u_i_j) / 2
        u_iMh_j = (u_iM1_j + u_i_j) / 2
        u_i_jMh = (u_i_jM1 + u_i_j) / 2

        v_iMh_jP1 = (v_i_jP1 + v_iM1_jP1) / 2
        v_iMh_j = (v_i_j + v_iM1_j) / 2

        # final expression 
        non_linear_parameters_x = np.dot((np.dot(u_iPh_j, u_iPh_j) - np.dot(u_iMh_j, u_iMh_j)),
                                         self.__delta_y) + \
                                  np.dot((np.dot(u_i_jPh, v_iMh_jP1) - np.dot(u_i_jMh, v_iMh_j)),
                                         self.__delta_x)

        return non_linear_parameters_x

    def __non_linear_parameters_y(self):
        # P = plus  M = minus h = half
        # find the matrices without the unneeded columns and rows
        u_iP1_jM1 = self.__u_matrix[2:, :-2]
        u_iP1_j = self.__u_matrix[2:, 1:-1]
        u_i_jM1 = self.__u_matrix[1:-1, :-2]
        u_i_j = self.__u_matrix[1:-1, 1:-1]

        v_i_j = self.__v_matrix[1:-1, 1:-1]
        v_iP1_j = self.__v_matrix[2:, 1:-1]
        v_i_jP1 = self.__v_matrix[1:-1, 2:]
        v_iM1_j = self.__v_matrix[:-2, 1:-1]
        v_i_jM1 = self.__v_matrix[1:-1, :-2]

        # find matrices with half indexes
        u_iP1_jMh = (u_iP1_jM1 + u_iP1_j) / 2
        u_i_jMh = (u_i_jM1 + u_i_j) / 2

        v_iPh_j = (v_i_j + v_iP1_j) / 2
        v_iMh_j = (v_i_j + v_iM1_j) / 2
        v_i_jPh = (v_i_j + v_i_jP1) / 2
        v_i_jMh = (v_i_j + v_i_jM1) / 2

        # final expression 
        non_linear_parameters_y = np.dot((np.dot(u_iP1_jMh, v_iPh_j) - np.dot(u_i_jMh, v_iMh_j)),
                                         self.__delta_y) + \
                                  np.dot((np.dot(v_i_jPh, v_i_jPh) - np.dot(v_i_jMh, v_i_jMh)),
                                         self.__delta_x)

        return non_linear_parameters_y
