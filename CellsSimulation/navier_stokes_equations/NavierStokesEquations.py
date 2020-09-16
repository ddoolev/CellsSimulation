import numpy as np
from LaplaceOperator import LaplaceOperator
import Constants as C
import scipy.sparse as sparse
from Boundaries import Boundaries
import matplotlib.pyplot as plt
import enum
from general_enums import Fields, Delta, Orientation


class BoundaryConditionsType(enum.Enum):
    dirichlet = 0
    neumann = 1


class WarningsStrings:
    neumann_pressure = "Should not use pressure values when in Neumann boundary condition type"


class NavierStokesEquations:
    __boundaries: Boundaries

    def __init__(self, field_matrix, delta_matrix, boundaries,
                 boundary_conditions_type=BoundaryConditionsType.neumann):
        self.__field_matrix = field_matrix
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
                                 self.__field_matrix[Fields.u]
        right_side_predicted_v = self.__pressure_terms_predicted_v() + \
                                 self.__non_linear_parameters_y() + \
                                 self.__field_matrix[Fields.v]
        predicted_u = self.__laplace_operator_velocity.solve(right_side_predicted_u)
        predicted_v = self.__laplace_operator_velocity.solve(right_side_predicted_v)

        # calculate p prime
        right_side_p_prime = -(self.__divergence_x(predicted_u, Fields.u) +
                               self.__divergence_y(predicted_v, Fields.v)) / self.__delta_t
        p_prime = self.__laplace_operator_p_prime.solve(right_side_p_prime)

        # calculate the new fields
        self.__field_matrix[Fields.p] = p_prime + self.__field_matrix[Fields.p]
        self.__field_matrix[Fields.u] = self.__field_matrix[Fields.u] - \
                                        np.dot(self.__divergence_x(p_prime, Fields.p), self.__delta_t)
        self.__field_matrix[Fields.v] = self.__field_matrix[Fields.v] - \
                                        np.dot(self.__divergence_y(p_prime, Fields.p), self.__delta_t)

    def __non_linear_parameters_x(self):
        # P = plus  M = minus h = half
        # find the matrices without the unneeded columns and rows
        u_iP1_j = self.__field_matrix[Fields.u][1:-1, 2:]
        u_i_j = self.__field_matrix[Fields.u][1:-1, 1:-1]
        u_iM1_j = self.__field_matrix[Fields.u][1:-1, :-2]
        u_i_jP1 = self.__field_matrix[Fields.u][2:, 1:-1]
        u_i_jM1 = self.__field_matrix[Fields.u][:-2, 1:-1]

        v_i_jP1 = self.__field_matrix[Fields.v][2:, 1:-1]
        v_iM1_jP1 = self.__field_matrix[Fields.v][2:, :-2]
        v_i_j = self.__field_matrix[Fields.v][1:-1, 1:-1]
        v_iM1_j = self.__field_matrix[Fields.v][1:-1, :-2]

        # find matrices with half indexes
        u_iPh_j = (u_iP1_j + u_i_j) / 2
        u_i_jPh = (u_i_jP1 + u_i_j) / 2
        u_iMh_j = (u_iM1_j + u_i_j) / 2
        u_i_jMh = (u_i_jM1 + u_i_j) / 2

        v_iMh_jP1 = (v_i_jP1 + v_iM1_jP1) / 2
        v_iMh_j = (v_i_j + v_iM1_j) / 2

        # final expression
        delta_y_multiplication = np.multiply(u_iPh_j, u_iPh_j)
        delta_y_multiplication -= np.multiply(u_iMh_j, u_iMh_j)
        delta_y_multiplication = delta_y_multiplication.T
        delta_x_multiplication = np.multiply(u_i_jPh, v_iMh_jP1)
        delta_x_multiplication -= np.multiply(u_i_jMh, v_iMh_j)
        non_linear_parameters_x = np.multiply(delta_y_multiplication, self.__delta_matrix[Delta.y]).T + \
                                  np.multiply(delta_x_multiplication, self.__delta_matrix[Delta.x])

        return self.__add_all_boundaries(non_linear_parameters_x, Fields.u)

    def __non_linear_parameters_y(self):
        # P = plus  M = minus h = half
        # find the matrices without the unneeded columns and rows
        u_iP1_jM1 = self.__field_matrix[Fields.u][:-2, 2:]
        u_iP1_j = self.__field_matrix[Fields.u][1:-1, 2:]
        u_i_jM1 = self.__field_matrix[Fields.u][:-2, 1:-1]
        u_i_j = self.__field_matrix[Fields.u][1:-1, 1:-1]

        v_i_j = self.__field_matrix[Fields.v][1:-1, 1:-1]
        v_iP1_j = self.__field_matrix[Fields.v][1:-1, 2:]
        v_i_jP1 = self.__field_matrix[Fields.v][2:, 1:-1]
        v_iM1_j = self.__field_matrix[Fields.v][1:-1, :-2]
        v_i_jM1 = self.__field_matrix[Fields.v][:-2, 1:-1]

        # find matrices with half indexes
        u_iP1_jMh = (u_iP1_jM1 + u_iP1_j) / 2
        u_i_jMh = (u_i_jM1 + u_i_j) / 2

        v_iPh_j = (v_i_j + v_iP1_j) / 2
        v_iMh_j = (v_i_j + v_iM1_j) / 2
        v_i_jPh = (v_i_j + v_i_jP1) / 2
        v_i_jMh = (v_i_j + v_i_jM1) / 2

        # final expression
        non_linear_parameters_y = np.dot((np.dot(u_iP1_jMh, v_iPh_j) - np.dot(u_i_jMh, v_iMh_j)),
                                         self.__delta_matrix[Delta.y]) + \
                                  np.dot((np.dot(v_i_jPh, v_i_jPh) - np.dot(v_i_jMh, v_i_jMh)),
                                         self.__delta_matrix[Delta.x])

        return self.__add_all_boundaries(non_linear_parameters_y, Fields.v)

    def __pressure_terms_predicted_u(self):
        # P = plus
        p_matrix_top = self.__boundaries.get_boundary(Orientation.top, Fields.p)
        p_matrix_bottom = self.__boundaries.get_boundary(Orientation.bottom, Fields.p)
        p_matrix_top_bottom_boundaries = \
            np.concatenate(([p_matrix_bottom], self.__field_matrix[Fields.p], [p_matrix_top]), axis=0)
        p_iP1_j = p_matrix_top_bottom_boundaries[:, 1:]
        p_i_j = p_matrix_top_bottom_boundaries[:, :-1]

        results = p_iP1_j - p_i_j
        delta_y_with_top_bottom_boundaries = \
            np.concatenate(([1], self.__delta_matrix[Delta.y], [1]))
        results = np.multiply(results.T, delta_y_with_top_bottom_boundaries).T
        results = self.__add_left_boundary(results, Fields.u)
        results = self.__add_right_boundary(results, Fields.u)
        return results

    def __pressure_terms_predicted_v(self):
        # P = plus
        p_matrix_left = np.array(self.__boundaries.get_boundary(Orientation.left, Fields.p)).T
        p_matrix_right = np.array(self.__boundaries.get_boundary(Orientation.right, Fields.p)).T
        p_matrix_left_right_boundaries = \
            np.concatenate((p_matrix_left, self.__field_matrix[Fields.p], p_matrix_right), axis=1)
        p_i_jP1 = p_matrix_left_right_boundaries[1:, :]
        p_i_j = p_matrix_left_right_boundaries[:-1, :]

        results = p_i_jP1 - p_i_j
        delta_x_with_left_right_boundaries = \
            np.concatenate(([1], self.__delta_matrix[Delta.x], [1]))
        results = np.multiply(results, delta_x_with_left_right_boundaries)
        results = self.__add_top_boundary(results, Fields.u)
        results = self.__add_bottom_boundary(results, Fields.u)
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

    def __add_left_boundary(self, matrix, field):
        if field == Fields.p and self.__boundary_conditions_type == BoundaryConditionsType.neumann:
            left_boundary = np.concatenate(([0], matrix[0], [0]))
        else:
            left_boundary = self.__boundaries.get_boundary(Orientation.left, field)
            left_boundary = np.concatenate(([0], left_boundary, [0]), axis=0)
        left_boundary = np.array([left_boundary]).T
        return np.append(left_boundary, matrix, axis=1)

    def __add_right_boundary(self, matrix, field):
        if field == Fields.p and self.boundary_conditions_type == BoundaryConditionsType.neumann:
            right_boundary = np.concatenate(([0], matrix[-1], [0]))
        else:
            right_boundary = self.__boundaries.get_boundary(Orientation.right, field)
            right_boundary = np.concatenate(([0], right_boundary, [0]), axis=0)
        right_boundary = np.array([right_boundary]).T
        return np.append(matrix, right_boundary, axis=1)

    def __add_bottom_boundary(self, matrix, field):
        if field == Fields.p and self.boundary_conditions_type == BoundaryConditionsType.neumann:
            bottom_boundary = np.concatenate(([0], matrix.T[0], [0]))
        else:
            bottom_boundary = self.__boundaries.get_boundary(Orientation.bottom, field)
            bottom_boundary = np.concatenate(([0], bottom_boundary, [0]), axis=0)
        return np.append([bottom_boundary], matrix, axis=0)

    def __add_top_boundary(self, matrix, field):
        if field == Fields.p and self.boundary_conditions_type == BoundaryConditionsType.neumann:
            top_boundary = np.concatenate(([0], matrix.T[-1], [0]))
        else:
            top_boundary = self.__boundaries.get_boundary(Orientation.top, field)
            top_boundary = np.concatenate(([0], top_boundary, [0]), axis=0)
        return np.append(matrix, [top_boundary], axis=0)

    def __add_all_boundaries(self, matrix, field):
        if field == Fields.p and self.boundary_conditions_type == BoundaryConditionsType.neumann:
            left = np.array([matrix[0]]).T
            right = np.array([matrix[-1]]).T
            bottom = [np.concatenate(([0], matrix[0].T, [0]), axis=0)]
            top = [np.concatenate(([0], matrix[-1].T, [0]), axis=0)]
        else:
            left = np.array([self.__boundaries.get_boundary(Orientation.left, field)]).T
            right = np.array([self.__boundaries.get_boundary(Orientation.right, field)]).T
            bottom = [np.concatenate(([0], self.__boundaries.get_boundary(Orientation.bottom, field), [0]), axis=0)]
            top = [np.concatenate(([0], self.__boundaries.get_boundary(Orientation.top, field), [0]), axis=0)]

        matrix = np.concatenate((left, matrix, right), axis=1)
        matrix = np.concatenate((bottom, matrix, top), axis=0)
        return matrix

    def __add_top_bottom_boundaries(self, matrix, field, with_left_right_boundaries=False):
        if with_left_right_boundaries:
            top = [np.concatenate(([0], self.__boundaries.get_boundary(Orientation.top, field), [0]), axis=0)]
            bottom = [np.concatenate(([0], self.__boundaries.get_boundary(Orientation.bottom, field), [0]), axis=0)]
        else:
            top = [self.__boundaries.get_boundary(Orientation.top, field)]
            bottom = [self.__boundaries.get_boundary(Orientation.bottom, field)]
        matrix = np.concatenate((bottom, matrix, top), axis=0)
        return matrix

    def __add_left_right_boundaries(self, matrix, field, with_top_bottom_boundaries=False):
        if with_top_bottom_boundaries:
            left = [np.concatenate(([0], self.__boundaries.get_boundary(Orientation.left, field), [0]), axis=0)]
            right = [np.concatenate(([0], self.__boundaries.get_boundary(Orientation.right, field), [0]), axis=0)]
        else:
            left = [self.__boundaries.get_boundary(Orientation.top, field)]
            right = [self.__boundaries.get_boundary(Orientation.right, field)]
        left = np.array([left]).T
        right = np.array([right]).T
        matrix = np.concatenate((left, matrix, right), axis=1)
        return matrix

    ###################################### On Index Fields

    def __get_index_u_matrix(self):
        left_boundary = np.array([self.__boundaries.get_boundary(Orientation.left, Fields.u)]).T
        right_boundary = np.array([self.__boundaries.get_boundary(Orientation.right, Fields.u)]).T
        index_u_matrix = np.concatenate((left_boundary, self.__field_matrix[Fields.u], right_boundary), axis=1)
        index_u_matrix = (index_u_matrix[1:, :] + index_u_matrix[:-1, :]) / 2
        index_u_matrix = self.__add_bottom_boundary(index_u_matrix, Fields.u)
        index_u_matrix = self.__add_top_boundary(index_u_matrix, Fields.u)
        return index_u_matrix

    def __get_index_v_matrix(self):
        bottom_boundary = [self.__boundaries.get_boundary(Orientation.bottom, Fields.v)]
        top_boundary = [self.__boundaries.get_boundary(Orientation.top, Fields.v)]
        index_v_matrix = np.concatenate((bottom_boundary, self.__field_matrix[Fields.v], top_boundary), axis=0)
        index_v_matrix = (index_v_matrix[:, 1:] + index_v_matrix[:, :-1]) / 2
        index_v_matrix = self.__add_left_boundary(index_v_matrix, Fields.v)
        index_v_matrix = self.__add_right_boundary(index_v_matrix, Fields.v)
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
