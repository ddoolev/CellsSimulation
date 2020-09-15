import numpy as np
from LaplaceOperator import LaplaceOperator
import Constants as C
import scipy.sparse as sparse
from Boundaries import Boundaries, Orientation
import matplotlib.pyplot as plt
from general_enums import Fields


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
        right_side_p_prime = -(self.__divergence_x(predicted_u, Fields.u) +
                               self.__divergence_y(predicted_v, Fields.v)) / self.__delta_t
        p_prime = self.__laplace_operator_p_prime.solve(right_side_p_prime)

        # calculate the new fields
        self.__p_matrix = p_prime + self.__p_matrix
        self.__u_matrix = self.__u_matrix - \
                          np.dot(self.__divergence_x(p_prime, Fields.p), self.__delta_t)
        self.__v_matrix = self.__v_matrix - \
                          np.dot(self.__divergence_y(p_prime, Fields.p), self.__delta_t)

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

        return self.__boundaries.add_boundaries_all(non_linear_parameters_x, Fields.u)

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

        return self.__boundaries.add_boundaries_all(non_linear_parameters_y, Fields.v)

    def __pressure_terms_x(self):
        # P = plus
        p_i_jP1 = self.__p_matrix[:, 2:-1]
        p_i_j = self.__p_matrix[:, 1:-2]

        results = p_i_jP1 - p_i_j
        results = np.dot(results, self.__delta_y)
        results = self.__boundaries.add_boundaries_left(results, Fields.u)
        results = self.__boundaries.add_boundaries_right(results, Fields.u)
        return results

    def __pressure_terms_y(self):
        # P = plus
        p_iP1_j = self.__p_matrix[2:-1, :]
        p_i_j = self.__p_matrix[1:-2, :]

        results = p_iP1_j - p_i_j
        results = np.dot(results.transpose(), self.__delta_x).transpose()
        results = self.__boundaries.add_boundaries_left(results, Fields.v)
        results = self.__boundaries.add_boundaries_right(results, Fields.v)
        return results

    def __divergence_x(self, matrix, field):
        # M = minus
        matrix_iP1_j = matrix[:, 1:]
        matrix_i_j = matrix[:, :-1]
        results = (matrix_iP1_j - matrix_i_j) / self.__delta_x
        results = self.__boundaries.add_boundaries_left(results, field)
        results = self.__boundaries.add_boundaries_right(results, field)
        return results

    def __divergence_y(self, matrix, field):
        # M = minus
        matrix_i_jP1 = matrix[1:, :]
        matrix_i_j = matrix[:-1, :]
        results = (matrix_i_jP1 - matrix_i_j) / self.__delta_x
        results = self.__boundaries.add_boundaries_bottom(results, field)
        results = self.__boundaries.add_boundaries_top(results, field)
        return results

    ###################################### On Index Fields

    def __get_index_u_matrix(self):
        left_boundary = np.array([self.__boundaries.get_boundary(Orientation.left, Fields.u)]).T
        right_boundary = np.array([self.__boundaries.get_boundary(Orientation.right, Fields.u)]).T
        index_u_matrix = np.concatenate((left_boundary, self.__u_matrix, right_boundary), axis=1)
        index_u_matrix = (index_u_matrix[1:, :] + index_u_matrix[:-1, :])/2
        index_u_matrix = self.__boundaries.add_boundaries_bottom(index_u_matrix, Fields.u)
        index_u_matrix = self.__boundaries.add_boundaries_top(index_u_matrix, Fields.u)
        return index_u_matrix

    def __get_index_v_matrix(self):
        bottom_boundary = [self.__boundaries.get_boundary(Orientation.bottom, Fields.v)]
        top_boundary = [self.__boundaries.get_boundary(Orientation.top, Fields.v)]
        index_v_matrix = np.concatenate((bottom_boundary, self.__v_matrix, top_boundary), axis=0)
        index_v_matrix = (index_v_matrix[:, 1:] + index_v_matrix[:, :-1])/2
        index_v_matrix = self.__boundaries.add_boundaries_left(index_v_matrix, Fields.v)
        index_v_matrix = self.__boundaries.add_boundaries_right(index_v_matrix, Fields.v)
        return index_v_matrix

    ###################################### Data options

    def quiver(self):
        x = np.append([0],  self.__delta_x).cumsum()
        y = np.append([0],  self.__delta_y).cumsum()
        xx, yy = np.meshgrid(x, y)

        index_u_matrix = self.__get_index_u_matrix()
        index_v_matrix = self.__get_index_v_matrix()
        plt.quiver(xx, yy, index_u_matrix, index_v_matrix)