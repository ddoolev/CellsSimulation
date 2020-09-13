import numpy as np
from LaplaceOperator import LaplaceOperator
import Constants as C
import scipy.sparse as sparse
from Boundaries import Boundaries, BoundaryConditionsType
import enum
import matplotlib.pyplot as plt
from enums import Fields


class ORIENTATION(enum.Enum):
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3


class NavierStokesEquations:

    __boundaries: Boundaries

    def __init__(self, p_matrix, v_matrix, u_matrix, delta_x, delta_y, boundaries):
        self.__p_matrix = p_matrix
        self.__v_matrix = v_matrix
        self.__u_matrix = u_matrix
        self.__delta_x = delta_x
        self.__delta_y = delta_y
        self.__boundaries = boundaries
        self.__delta_t = C.TIME_STEP

        self.__laplace_operator_p_prime = LaplaceOperator(delta_x, delta_y)

        self.__laplace_operator_velocity = LaplaceOperator(delta_x, delta_y)
        self.__laplace_operator_velocity.multiply_operators_matrix(-1 / C.Re)
        matrix_side_length = (len(delta_x) + 1) * (len(delta_y) + 1)
        identity_matrix = sparse.diags(np.full(matrix_side_length, 1))
        self.__laplace_operator_velocity = self.__laplace_operator_velocity.add_to_operators_matrix(identity_matrix)

    def next_step(self):
        # calculate predicted u and v
        right_side_predicted_u = self.__pressure_terms_x() + \
                                 self.__non_linear_parameters_x() + \
                                 self.__u_matrix
        right_side_predicted_v = self.__pressure_terms_y() + \
                                 self.__non_linear_parameters_y() + \
                                 self.__v_matrix
        predicted_u = self.__laplace_operator_velocity.solve(right_side_predicted_u)
        predicted_v = self.__laplace_operator_velocity.solve(right_side_predicted_v)

        # calculate p prime
        right_side_p_prime = -(self.__divergence_x(predicted_u, Fields.U) +
                               self.__divergence_y(predicted_v, Fields.V)) / self.__delta_t
        p_prime = self.__laplace_operator_p_prime.solve(right_side_p_prime)

        # calculate the new fields
        self.__p_matrix = p_prime + self.__p_matrix
        self.__u_matrix = self.__u_matrix - \
                          np.dot(self.__divergence_x(p_prime, Fields.P), self.__delta_t)
        self.__v_matrix = self.__v_matrix - \
                          np.dot(self.__divergence_y(p_prime, Fields.P), self.__delta_t)

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

        return self.__add_boundaries_all(non_linear_parameters_x, Fields.U)

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

        return self.__add_boundaries_all(non_linear_parameters_y, Fields.V)

    def __pressure_terms_x(self):
        # P = plus
        p_i_jP1 = self.__p_matrix[:, 2:-1]
        p_i_j = self.__p_matrix[:, 1:-2]

        results = p_i_jP1 - p_i_j
        results = np.dot(results, self.__delta_y)
        results = self.__add_boundaries_left(results, Fields.U)
        results = self.__add_boundaries_right(results, Fields.U)
        return results

    def __pressure_terms_y(self):
        # P = plus
        p_iP1_j = self.__p_matrix[2:-1, :]
        p_i_j = self.__p_matrix[1:-2, :]

        results = p_iP1_j - p_i_j
        results = np.dot(results.transpose(), self.__delta_x).transpose()
        results = self.__add_boundaries_left(results, Fields.V)
        results = self.__add_boundaries_right(results, Fields.V)
        return results

    def __divergence_x(self, matrix, field):
        # M = minus
        matrix_iP1_j = matrix[:, 1:]
        matrix_i_j = matrix[:, :-1]
        results = (matrix_iP1_j - matrix_i_j) / self.__delta_x
        results = self.__add_boundaries_left(results, field)
        results = self.__add_boundaries_right(results, field)
        return results

    def __divergence_y(self, matrix, field):
        # M = minus
        matrix_i_jP1 = matrix[1:, :]
        matrix_i_j = matrix[:-1, :]
        results = (matrix_i_jP1 - matrix_i_j) / self.__delta_x
        results = self.__add_boundaries_top(results, field)
        results = self.__add_boundaries_bottom(results, field)
        return results

    ###################################### Boundaries

    def __add_boundaries_left(self, matrix, field):
        if field == Fields.P and self.__boundaries.boundary_conditions_type == BoundaryConditionsType.NUEMANN:
            left_boundary = np.concatenate(([0], matrix[0], [0]), axis=0)
        else:
            left_boundary = np.concatenate(([0], self.__boundaries.get_left(field), [0]), axis=0)
        return np.concatenate((left_boundary.T, matrix), axis=1)

    def __add_boundaries_right(self, matrix, field):
        if field == Fields.P and self.__boundaries.boundary_conditions_type == BoundaryConditionsType.NUEMANN:
            right_boundary = np.concatenate(([0], matrix[-1], [0]), axis=0)
        else:
            right_boundary = np.concatenate(([0], self.__boundaries.get_right(field), [0]), axis=0)
        return np.concatenate((matrix, right_boundary.T), axis=1)

    def __add_boundaries_top(self, matrix, field):
        if field == Fields.P and self.__boundaries.boundary_conditions_type == BoundaryConditionsType.NUEMANN:
            top_boundary = np.concatenate(([0], matrix.T[0], [0]), axis=0)
        else:
            top_boundary = np.concatenate(([0], self.__boundaries.get_top(field), [0]), axis=0)
        return np.concatenate((top_boundary, matrix), axis=0)

    def __add_boundaries_bottom(self, matrix, field):
        if field == Fields.P and self.__boundaries.boundary_conditions_type == BoundaryConditionsType.NUEMANN:
            bottom_boundary = np.concatenate(([0], matrix.T[-1], [0]), axis=0)
        else:
            bottom_boundary = np.concatenate(([0], self.__boundaries.get_bottom(field), [0]), axis=0)
        return np.concatenate((matrix, bottom_boundary), axis=0)

    def __add_boundaries_all(self, matrix, field):
        if field == Fields.P and self.__boundaries.boundary_conditions_type == BoundaryConditionsType.NUEMANN:
            left_boundary = np.array([matrix[0]]).T
            right_boundary = np.array([matrix[-1]]).T
            top_boundary = np.concatenate(([0], matrix[0].T, [0]), axis=0)
            bottom_boundary = np.concatenate(([0], matrix[-1].T, [0]), axis=0)
        else:
            left_boundary = np.array([self.__boundaries.get_left(field)]).T
            right_boundary = np.array([self.__boundaries.get_right(field)]).T
            top_boundary = np.concatenate(([0], self.__boundaries.get_top(field), [0]), axis=0)
            bottom_boundary = np.concatenate(([0], self.__boundaries.get_bottom(field), [0]), axis=0)
        top_boundary = np.array([top_boundary])
        bottom_boundary = np.array([bottom_boundary])

        matrix = np.concatenate((left_boundary, matrix, right_boundary), axis=1)
        matrix = np.concatenate((top_boundary, matrix, bottom_boundary), axis=0)
        return matrix

    ###################################### Data options

    def quiver(self):
        x = np.concatenate(([0], self.__delta_x)).cumsum()
        y = np.concatenate(([0], self.__delta_y)).cumsum()
        xx, yy = np.meshgrid(x, y)
        full_u_matrix = self.__add_boundaries_all(self.__u_matrix, Fields.U)
        full_v_matrix = self.__add_boundaries_all(self.__v_matrix, Fields.V)
        plt.quiver(xx, yy, full_u_matrix, full_v_matrix)


