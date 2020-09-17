import numpy as np
from LaplaceOperator import LaplaceOperator
import Constants as C
import scipy.sparse as sparse
from Boundaries import Boundaries
import matplotlib.pyplot as plt
import enum
from general_enums import Field, Delta, Orientation


class BoundaryConditionsType(enum.Enum):
    dirichlet = 0
    neumann = 1


class WarningsStrings:
    neumann_pressure = "Should not use pressure values when in Neumann boundary condition type"


class NavierStokesEquations:
    __boundaries: Boundaries

    def __init__(self, field_matrix, delta_matrix, boundaries,
                 boundary_conditions_type=BoundaryConditionsType.neumann):
        self.__fields_matrix = field_matrix
        self.__delta_matrix = delta_matrix
        self.__boundaries = boundaries
        self.__delta_t = C.TIME_STEP
        # decide the pressure boundaries type: Dirichlet or Neumann
        self.__boundary_conditions_type = boundary_conditions_type

        # make the laplacian operators
        self.__laplace_operator_p_prime = LaplaceOperator(delta_matrix[Delta.x], delta_matrix[Delta.y])

        self.__laplace_operator_velocity = LaplaceOperator(delta_matrix[Delta.x], delta_matrix[Delta.y])
        self.__laplace_operator_velocity.multiply_operators_matrix(-1 / C.Re)
        matrix_side_length = (len(delta_matrix[Delta.x]) + 1) * (len(delta_matrix[Delta.y]) + 1)
        identity_matrix = sparse.diags(np.full(matrix_side_length, 1))
        self.__laplace_operator_velocity = self.__laplace_operator_velocity.add_to_operators_matrix(identity_matrix)

    def next_step(self):
        # calculate predicted u and v

        right_side_predicted_u = self.__pressure_terms_predicted_u() + \
                                 self.__non_linear_parameters_x() + \
                                 self.__fields_matrix[Field.u]

        right_side_predicted_v = self.__pressure_terms_predicted_v() + \
                                 self.__non_linear_parameters_y() + \
                                 self.__fields_matrix[Field.v]
        predicted_u = self.__laplace_operator_velocity.solve(right_side_predicted_u)
        predicted_v = self.__laplace_operator_velocity.solve(right_side_predicted_v)

        # calculate p prime
        right_side_p_prime = -(self.__divergence_x(predicted_u, Field.u) +
                               self.__divergence_y(predicted_v, Field.v)) / self.__delta_t
        p_prime = self.__laplace_operator_p_prime.solve(right_side_p_prime)

        # calculate the new fields
        self.__fields_matrix[Field.p] = p_prime + self.__fields_matrix[Field.p]
        self.__fields_matrix[Field.u] = self.__fields_matrix[Field.u] - \
                                        np.dot(self.__divergence_x(p_prime, Field.p), self.__delta_t)
        self.__fields_matrix[Field.v] = self.__fields_matrix[Field.v] - \
                                        np.dot(self.__divergence_y(p_prime, Field.p), self.__delta_t)

    def __non_linear_parameters_x(self):
        # P = plus  M = minus h = half
        u_matrix = self.__fields_matrix[Field.u]
        v_matrix = self.__fields_matrix[Field.v]

        # find the matrices without the unneeded columns and rows
        u_iP1_j = self.__add_right_boundary(u_matrix[:, 1:], Field.u)
        u_i_j = u_matrix
        u_iM1_j = self.__add_left_boundary(u_matrix[:, :-1], Field.u)
        u_i_jP1 = u_matrix[1:, :]
        u_i_jM1 = u_matrix[:-1, :]

        v_i_jP1 = self.__add_top_boundary(v_matrix, Field.v)[:, 1:]
        v_iM1_jP1 = self.__add_top_boundary(v_matrix, Field.v)[:, :-1]
        v_i_j = self.__add_bottom_boundary(v_matrix, Field.v)[:, 1:]
        v_iM1_j = self.__add_bottom_boundary(v_matrix, Field.v)[:, :-1]

        # find matrices with half indexes
        u_iPh_j = (u_iP1_j + u_i_j) / 2
        u_iMh_j = (u_iM1_j + u_i_j) / 2
        u_i_middle_j = (u_i_jP1 + u_i_jM1) / 2
        u_i_jPh = self.__add_top_boundary(u_i_middle_j, Field.u)
        u_i_jMh = self.__add_bottom_boundary(u_i_middle_j, Field.u)

        v_iMh_jP1 = (v_i_jP1 + v_iM1_jP1) / 2
        v_iMh_j = (v_i_j + v_iM1_j) / 2

        # final expression
        delta_y_multiplication = np.multiply(u_iPh_j, u_iPh_j)
        delta_y_multiplication -= np.multiply(u_iMh_j, u_iMh_j)
        delta_x_multiplication = np.multiply(u_i_jPh, v_iMh_jP1)
        delta_x_multiplication -= np.multiply(u_i_jMh, v_iMh_j)
        delta_x = self.__delta_matrix[Delta.x]
        delta_x_middle = (delta_x[1:] + delta_x[:-1])/2
        non_linear_parameters_x = np.multiply(delta_y_multiplication.T, self.__delta_matrix[Delta.y]).T + \
                                  np.multiply(delta_x_multiplication, delta_x_middle)

        return non_linear_parameters_x

    def __non_linear_parameters_y(self):
        # P = plus  M = minus h = half
        u_matrix = self.__fields_matrix[Field.u]
        v_matrix = self.__fields_matrix[Field.v]

        # find the matrices without the unneeded columns and rows
        u_iP1_jM1 = self.__add_right_boundary(u_matrix, Field.u)[:-1, :]
        u_iP1_j = self.__add_right_boundary(u_matrix, Field.u)[1:, :]
        u_i_jM1 = self.__add_left_boundary(u_matrix, Field.u)[:-1, :]
        u_i_j = self.__add_left_boundary(u_matrix, Field.u)[1:, :]

        v_i_jP1 = self.__add_top_boundary(v_matrix[1:, :], Field.v)
        v_i_j = v_matrix
        v_i_jM1 = self.__add_bottom_boundary(v_matrix[:-1, :], Field.v)
        v_iP1_j = v_matrix[:, 1:]
        v_iM1_j = v_matrix[:, :-1]

        # find matrices with half indexes
        u_iP1_jMh = (u_iP1_jM1 + u_iP1_j) / 2
        u_i_jMh = (u_i_jM1 + u_i_j) / 2

        v_i_jPh = (v_i_j + v_i_jP1) / 2
        v_i_jMh = (v_i_j + v_i_jM1) / 2
        v_middle_i_j = (v_iP1_j + v_iM1_j) / 2
        v_iPh_j = self.__add_right_boundary(v_middle_i_j, Field.v)
        v_iMh_j = self.__add_left_boundary(v_middle_i_j, Field.v)

        # final expression
        delta_x_multiplication = np.multiply(v_i_jPh, v_i_jPh)
        delta_x_multiplication -= np.multiply(v_i_jMh, v_i_jMh)
        delta_y_multiplication = np.multiply(u_iP1_jMh, v_iPh_j)
        delta_y_multiplication -= np.multiply(u_i_jMh, v_iMh_j)
        delta_y = self.__delta_matrix[Delta.y]
        delta_y_middle = (delta_y[1:] + delta_y[:-1])/2
        non_linear_parameters_x = np.multiply(delta_y_multiplication.T, delta_y_middle).T + \
                                  np.multiply(delta_x_multiplication, self.__delta_matrix[Delta.x])

        return non_linear_parameters_x

    def __pressure_terms_predicted_u(self):
        # P = plus
        p_iP1_j = self.__fields_matrix[Field.p][:, 1:]
        p_i_j = self.__fields_matrix[Field.p][:, :-1]

        results = p_iP1_j - p_i_j
        results = np.multiply(results.T, self.__delta_matrix[Delta.y]).T
        return results

    def __pressure_terms_predicted_v(self):
        # P = plus
        p_i_jP1 = self.__fields_matrix[Field.p][1:, :]
        p_i_j = self.__fields_matrix[Field.p][:-1, :]
        results = p_i_jP1 - p_i_j

        results = np.multiply(results, self.__delta_matrix[Delta.x])
        return results

    def __divergence_x(self, matrix, field):
        # M = minus
        matrix_iP1_j = matrix[1:, :]
        matrix_i_j = matrix[:-1, :]
        results = (matrix_iP1_j - matrix_i_j) / self.__delta_matrix[Delta.x]
        results = self.__add_left_boundary(results, field)
        results = self.__add_right_boundary(results, field)
        return results

    def __divergence_y(self, matrix, field):
        # M = minus
        matrix_i_jP1 = matrix[:, 1:]
        matrix_i_j = matrix[:, :-1]
        results = (matrix_i_jP1 - matrix_i_j) / self.__delta_matrix[Delta.x]
        results = self.__add_bottom_boundary(results, field)
        results = self.__add_top_boundary(results, field)
        return results

    ###################################### Boundaries

    def __add_left_boundary(self, matrix, field, with_top_bottom_boundaries=False):
        if field == Field.p and self.__boundary_conditions_type == BoundaryConditionsType.neumann:
            left_boundary = matrix.T[-1]
        else:
            left_boundary = self.__boundaries.get_boundary(Orientation.left, field)

        if with_top_bottom_boundaries:
            left_boundary = np.concatenate(([0], left_boundary, [0]))

        left_boundary = np.array([left_boundary]).T
        return np.append(left_boundary, matrix, axis=1)

    def __add_right_boundary(self, matrix, field, with_top_bottom_boundaries=False):
        if field == Field.p and self.__boundary_conditions_type == BoundaryConditionsType.neumann:
            right_boundary = matrix.T[0]
        else:
            right_boundary = self.__boundaries.get_boundary(Orientation.right, field)

        if with_top_bottom_boundaries:
            right_boundary = np.concatenate(([0], right_boundary, [0]))

        right_boundary = np.array([right_boundary]).T
        return np.append(matrix, right_boundary, axis=1)

    def __add_bottom_boundary(self, matrix, field, with_left_right_boundaries=False):
        if field == Field.p and self.__boundary_conditions_type == BoundaryConditionsType.neumann:
            bottom_boundary = matrix[0]
        else:
            bottom_boundary = self.__boundaries.get_boundary(Orientation.bottom, field)

        if with_left_right_boundaries:
            bottom_boundary = np.concatenate(([0], bottom_boundary, [0]))

        return np.append([bottom_boundary], matrix, axis=0)

    def __add_top_boundary(self, matrix, field, with_left_right_boundaries=False):
        if field == Field.p and self.__boundary_conditions_type == BoundaryConditionsType.neumann:
            top_boundary = matrix[-1]
        else:
            top_boundary = self.__boundaries.get_boundary(Orientation.top, field)

        if with_left_right_boundaries:
            top_boundary = np.concatenate(([0], top_boundary, [0]))

        return np.append(matrix, [top_boundary], axis=0)

    def __add_all_boundaries(self, matrix, field):
        matrix = self.__add_top_boundary(matrix, field, with_left_right_boundaries=False)
        matrix = self.__add_bottom_boundary(matrix, field, with_left_right_boundaries=False)
        matrix = self.__add_left_boundary(matrix, field, with_top_bottom_boundaries=True)
        matrix = self.__add_right_boundary(matrix, field, with_top_bottom_boundaries=True)
        return matrix

    ###################################### On Index Fields

    def __get_index_u_matrix(self):
        left_boundary = np.array([self.__boundaries.get_boundary(Orientation.left, Field.u)]).T
        right_boundary = np.array([self.__boundaries.get_boundary(Orientation.right, Field.u)]).T
        index_u_matrix = np.concatenate((left_boundary, self.__fields_matrix[Field.u], right_boundary), axis=1)
        index_u_matrix = (index_u_matrix[1:, :] + index_u_matrix[:-1, :]) / 2
        index_u_matrix = self.__add_bottom_boundary(index_u_matrix, Field.u, with_left_right_boundaries=True)
        index_u_matrix = self.__add_top_boundary(index_u_matrix, Field.u, with_left_right_boundaries=True)
        return index_u_matrix

    def __get_index_v_matrix(self):
        bottom_boundary = [self.__boundaries.get_boundary(Orientation.bottom, Field.v)]
        top_boundary = [self.__boundaries.get_boundary(Orientation.top, Field.v)]
        index_v_matrix = np.concatenate((bottom_boundary, self.__fields_matrix[Field.v], top_boundary), axis=0)
        index_v_matrix = (index_v_matrix[:, 1:] + index_v_matrix[:, :-1]) / 2
        index_v_matrix = self.__add_left_boundary(index_v_matrix, Field.v, with_top_bottom_boundaries=True)
        index_v_matrix = self.__add_right_boundary(index_v_matrix, Field.v, with_top_bottom_boundaries=True)
        return index_v_matrix

    ###################################### Data options

    def quiver(self):
        x = np.append([0], self.__delta_matrix[Delta.x]).cumsum()
        y = np.append([0], self.__delta_matrix[Delta.y]).cumsum()
        xx, yy = np.meshgrid(x, y)

        index_u_matrix = self.__get_index_u_matrix()
        index_v_matrix = self.__get_index_v_matrix()
        plt.quiver(xx, yy, index_u_matrix, index_v_matrix)

    ###################################### Setters and Getters

    @property
    def boundary_conditions_type(self):
        return self.__boundary_conditions_type

    @boundary_conditions_type.setter
    def boundary_conditions_type(self, boundary_conditions_type):
        self.__boundary_conditions_type = boundary_conditions_type
