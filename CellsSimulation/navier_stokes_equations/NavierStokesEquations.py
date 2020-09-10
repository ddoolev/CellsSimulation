import numpy as np
from LaplaceOperator import LaplaceOperator
import Constants as C
import scipy
from Boundaries import Boundaries
import enum


class ORIENTATION(enum.Enum):
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3


class FIELD(enum.Enum):
    U = 0
    V = 1
    P = 2


class NavierStokesEquations:

    __boundaries: Boundaries

    def __init__(self, p_matrix, v_matrix, u_matrix, delta_x, delta_y, boundaries):
        self.__p_matrix = p_matrix
        self.__v_matrix = v_matrix
        self.__u_matrix = u_matrix
        self.__delta_x = delta_x
        self.__delta_y = delta_y
        self.__boundaries = boundaries

        self.__laplace_operator_p_prime = LaplaceOperator(delta_x, delta_y)
        self.__laplace_operator_velocity = LaplaceOperator(delta_x, delta_y)
        self.__laplace_operator_velocity.multiply_operators_matrix(-1 / C.Re)
        identity_matrix = scipy.sparse.dia_matrix \
            (np.full([delta_x * delta_y], 1), shape=(delta_x * delta_y, delta_x * delta_y))
        self.__laplace_operator_velocity = self.__laplace_operator_velocity.add_to_operators_matrix(identity_matrix)

    def next_step(self):
        right_side_non_solinoidal_u = self.__pressure_terms_x() + \
                                      self.__non_linear_parameters_x() + \
                                      self.__u_matrix
        right_side_non_solinoidal_v = self.__pressure_terms_y() + \
                                      self.__non_linear_parameters_y() + \
                                      self.__v_matrix
        predicted_u = self.__laplace_operator_velocity.solve(right_side_non_solinoidal_u)
        predicted_v = self.__laplace_operator_velocity.solve(right_side_non_solinoidal_v)
        right_side_p_prime = (self.__divergence_x(predicted_u) +
                              (predicted_v[:, 1:] - predicted_u[:, :-1]) * self.__delta_x) / \
                              C.TIME_STEP
        p_prime = self.__laplace_operator_p_prime.solve(right_side_p_prime)


    def __non_linear_parameters_x(self):
        # P = plus  M = minus h = half
        # find the matrices without the unneeded columns and rows
        u_iP1_j = self.__u_matrix[2:, 1:-1]
        u_i_j = self.__u_matrix[1:-1, 1:-1]
        u_iM1_j = self.__u_matrix[:-2, 1:-1]
        u_i_jP1 = self.__u_matrix[1:-1, 2:]
        u_i_jM1 = self.__u_matrix[1:-1, :-2]

        v_i_jP1 = self.__v_matrix[1:-1, 2:]
        v_iM1_jP1 = self.__v_matrix[:-2, 2:]
        v_i_j = self.__v_matrix[1:-1, 1:-1]
        v_iM1_j = self.__v_matrix[:-2, 1:-1]

        # find matrices with half indexes
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

        return self.__add_boundaries_full(non_linear_parameters_x)

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

        return self.__add_boundaries_values(non_linear_parameters_y)

    def __pressure_terms_x(self):
        # P = plus
        p_i_jP1 = self.__p_matrix[1:-1, 1:]
        p_i_j = self.__p_matrix[1:-1, :-1]

        results = p_i_jP1 - p_i_j
        results = np.dot(results, self.__delta_y)
        return self.__add_boundaries_values(results)

    def __pressure_terms_y(self):
        # P = plus
        p_iP1_j = self.__p_matrix[1:, 1:-1]
        p_i_j = self.__p_matrix[:-1, 1:-1]

        results = p_iP1_j - p_i_j
        results = np.dot(results.transpose(), self.__delta_x).transpose()
        return self.__add_boundaries_values(results)

    def __divergence_x(self):
        # M = minus
        u_i_j = self.__u_matrix[1:, 1:]
        u_iM1_j = self.__u_matrix[:-1, 1:]
        v_i_j = self.__v_matrix[1:, 1:]
        v_i_jM1 = self.__v_matrix[1:, :-1]
        return (u_i_j - u_iM1_j + v_i_j - v_i_jM1) * self.__delta_y

    def __divergence_y(self):
        # M = minus
        u_i_j = self.__u_matrix[1:, 1:]
        u_iM1_j = self.__u_matrix[:-1, 1:]
        v_i_j = self.__v_matrix[1:, 1:]
        v_i_jM1 = self.__v_matrix[1:, :-1]
        return (u_i_j - u_iM1_j + v_i_j - v_i_jM1) * self.__delta_x

    ###################################### Boundaries

    def __add_boundaries_left(self, matrix):
        left_boundary = np.concatenate(([0], self.__boundaries.,[0]), axis=0)
        return np.concatenate((left_boundary.T, matrix), axis=1)

    def __add_boundaries_right(self, matrix):
        right_boundary = np.concatenate(([0], matrix[Boundaries.SIDE.RIGHT],[0]), axis=0)
        return np.concatenate((matrix, right_boundary.T), axis=1)

    def __add_boundaries_top(self, matrix):
        top_boundary = np.concatenate(([0], matrix[Boundaries.SIDE.TOP],[0]), axis=0)
        return np.concatenate((top_boundary, matrix), axis=0)

    def __add_boundaries_bottom(self, matrix):
        bottom_boundary = np.concatenate(([0], matrix[Boundaries.SIDE.BOTTOM],[0]), axis=0)
        return np.concatenate((bottom_boundary.T, matrix), axis=1)

    @staticmethod
    def __add_boundaries_all(self, matrix):
        matrix = np.concatenate((bottom_boundary.T, matrix), axis=1)
        bottom_boundary = np.concatenate(([0], matrix[Boundaries.SIDE.BOTTOM], [0]), axis=0)
        return np.concatenate((bottom_boundary.T, matrix), axis=1)

