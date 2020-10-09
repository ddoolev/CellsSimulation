import numpy as np
from LaplaceOperator import LaplaceOperator
import Constants as C
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from general_enums import Field, Orientation, BoundaryConditionsType
from Boundaries import Boundaries
import enum


class Information(enum.Flag):
    none = enum.auto()
    check_divergent = enum.auto()
    check_gradient_p_dot_u_vector = enum.auto()
    all = check_divergent | check_gradient_p_dot_u_vector


class NavierStokesEquations:

    def __init__(self, field_matrix, delta, boundaries,
                 boundary_conditions_type=BoundaryConditionsType.neumann,
                 information=Information.none):

        self.__fields_matrix = field_matrix
        self.__delta = delta
        self.__boundaries = boundaries
        self.__information = information
        # decide the pressure boundaries type: Dirichlet or Neumann
        self.__boundary_conditions_type = boundary_conditions_type

        # make the laplacian operators
        self.__create_p_prime_laplacian()
        self.__create_predicted_u_laplacian()
        self.__create_predicted_v_laplacian()

    ###################################### Create laplacians

    def __create_p_prime_laplacian(self):
        self.__laplace_operator_p_prime = LaplaceOperator(self.__delta, Field.p,
                                                          BoundaryConditionsType.neumann)

    def __create_predicted_u_laplacian(self):
        self.__laplace_operator_predicted_u = LaplaceOperator(self.__delta, Field.u,
                                                              BoundaryConditionsType.dirichlet)

        self.__laplace_operator_predicted_u.multiply_operators_matrix(1 / C.Re)

        # add delta_x*delta_y/delta_t
        u_matrix_side_length = (len(self.__delta.x) + 1) * (len(self.__delta.half_y) + 1)
        u_identity_matrix = np.full(u_matrix_side_length, 1)

        delta_x = Boundaries.remove_side(self.__delta.half_x, Orientation.all)
        delta_x = np.concatenate(([0], delta_x, [0]))
        delta_x = np.tile(delta_x, (len(self.__delta.y) + 2))
        delta_y = np.concatenate(([0], self.__delta.y, [0]))
        delta_y = np.repeat(delta_y, len(self.__delta.half_x))

        u_identity_matrix = np.multiply(u_identity_matrix, delta_x)
        u_identity_matrix = np.multiply(u_identity_matrix.T, delta_y).T
        u_identity_matrix = np.multiply(u_identity_matrix, 1/-C.DELTA_T)
        u_identity_matrix = sparse.spdiags(u_identity_matrix, [0],
                                           u_matrix_side_length, u_matrix_side_length, format='csr')

        self.__laplace_operator_predicted_u.add_matrix_to_operators_matrix(u_identity_matrix)

    def __create_predicted_v_laplacian(self):
        self.__laplace_operator_predicted_v = LaplaceOperator(self.__delta, Field.v,
                                                              BoundaryConditionsType.dirichlet)

        self.__laplace_operator_predicted_v.multiply_operators_matrix(1 / C.Re)

        # add delta_x*delta_y/delta_t
        v_matrix_side_length = (len(self.__delta.x) + 1) * (len(self.__delta.half_y) + 1)
        v_identity_matrix = np.full(v_matrix_side_length, 1)

        delta_x = np.concatenate(([0], self.__delta.x, [0]))
        delta_x = np.tile(delta_x, len(self.__delta.half_x))
        delta_y = Boundaries.remove_side(self.__delta.half_y, Orientation.all)
        delta_y = np.concatenate(([0], delta_y, [0]))
        delta_y = np.repeat(delta_y, (len(self.__delta.x) + 2))

        v_identity_matrix = np.multiply(v_identity_matrix, delta_x)
        v_identity_matrix = np.multiply(v_identity_matrix, delta_y)
        v_identity_matrix = np.multiply(v_identity_matrix, 1/-C.DELTA_T)
        v_identity_matrix = sparse.spdiags(v_identity_matrix, [0],
                                           v_matrix_side_length, v_matrix_side_length, format='csr')

        self.__laplace_operator_predicted_v.add_matrix_to_operators_matrix(v_identity_matrix)

    def next_step(self):
        predicted_u = self.__calculate_predicted_u()
        predicted_v = self.__calculate_predicted_v()
        p_prime = self.__calculate_p_prime(predicted_u, predicted_v)
        self.__calculate_new_fields(p_prime, predicted_u, predicted_v)
        if self.__information & Information.check_divergent:
            self.__check_divergence()
        if self.__information & Information.check_gradient_p_dot_u_vector:
            self.__check_gradient_p_dot_u_vector()
        self.__check_num_3()

    def __check_divergence(self):
        u_divergence_x = self.__divergence_x_field_u(self.__fields_matrix[Field.u])
        v_divergence_y = self.__divergence_y_field_v(self.__fields_matrix[Field.v])
        matrix = u_divergence_x + v_divergence_y
        print("Max divergence = ", matrix.max())

    def __check_gradient_p_dot_u_vector(self):
        # P = plus  M = minus h = half
        p_i_j = self.__fields_matrix[Field.p]
        p_iP1_j = Boundaries.remove_side(p_i_j, Orientation.left)
        p_iM1_j = Boundaries.remove_side(p_i_j, Orientation.right)
        p_i_jP1 = Boundaries.remove_side(p_i_j, Orientation.bottom)
        p_i_jM1 = Boundaries.remove_side(p_i_j, Orientation.top)

        p_middle_i_j = (p_iP1_j + p_iM1_j) / 2
        p_iPh_j = Boundaries.add_boundaries(p_middle_i_j, self.__boundaries, Field.p, Orientation.right)
        p_iMh_j = Boundaries.add_boundaries(p_middle_i_j, self.__boundaries, Field.p, Orientation.left)
        p_i_middle_j = (p_i_jP1 + p_i_jM1) / 2
        p_i_jPh = Boundaries.add_boundaries(p_i_middle_j, self.__boundaries, Field.p, Orientation.top)
        p_i_jMh = Boundaries.add_boundaries(p_i_middle_j, self.__boundaries, Field.p, Orientation.bottom)

        u_i_j = self.__fields_matrix[Field.u]
        u_iM1_j = Boundaries.add_boundaries(u_i_j, self.__boundaries, Field.u, Orientation.left)
        u_i_j = Boundaries.add_boundaries(u_i_j, self.__boundaries, Field.u, Orientation.right)
        v_i_j = self.__fields_matrix[Field.v]
        v_i_jM1 = Boundaries.add_boundaries(v_i_j, self.__boundaries, Field.v, Orientation.bottom)
        v_i_j = Boundaries.add_boundaries(v_i_j, self.__boundaries, Field.v, Orientation.top)

        results = np.multiply(p_iPh_j, u_i_j)
        results -= np.multiply(p_iMh_j, u_iM1_j)
        results += np.multiply(p_i_jPh, v_i_j)
        results -= np.multiply(p_i_jMh, v_i_jM1)

        print("Max gradient p dot u vector = ", results.max())

    def __calculate_predicted_u(self):
        gradient_p = self.__gradient_p_predicted_u()
        derivative_t = self.__derivative_t_for_momentum_conservation_u()
        non_linear = self.__non_linear_parameters_x()
        right_side_predicted_u = gradient_p - derivative_t + non_linear
        right_side_predicted_u = Boundaries.add_boundaries(right_side_predicted_u, self.__boundaries,
                                                           Field.u, Orientation.all)
        predicted_u = self.__laplace_operator_predicted_u.solve(right_side_predicted_u)
        return predicted_u

    def __calculate_predicted_v(self):
        gradient_p = self.__gradient_p_predicted_v()
        derivative_t = self.__derivative_t_for_momentum_conservation_v()
        non_linear_parameters = self.__non_linear_parameters_y()
        right_side_predicted_v = gradient_p - derivative_t + non_linear_parameters
        right_side_predicted_v = Boundaries.add_boundaries(right_side_predicted_v, self.__boundaries,
                                                           Field.v, Orientation.all)
        predicted_v = self.__laplace_operator_predicted_v.solve(right_side_predicted_v)
        return predicted_v

    def __calculate_p_prime(self, predicted_u, predicted_v):
        predicted_u = Boundaries.remove_side(predicted_u, Orientation.all)
        predicted_v = Boundaries.remove_side(predicted_v, Orientation.all)
        divergence_x_predicted_u = self.__divergence_x_field_u(predicted_u)
        divergence_y_predicted_v = self.__divergence_y_field_v(predicted_v)
        right_side_p_prime = divergence_x_predicted_u + divergence_y_predicted_v
        right_side_p_prime /= self.__delta.t
        right_side_p_prime[0][0] = 0
        right_side_p_prime = Boundaries.add_boundaries(right_side_p_prime, self.__boundaries,
                                                       Field.p, Orientation.all)
        p_prime = self.__laplace_operator_p_prime.solve(right_side_p_prime)
        return p_prime

    def __calculate_new_fields(self, p_prime, predicted_u, predicted_v):
        predicted_u = Boundaries.remove_side(predicted_u, Orientation.all)
        predicted_v = Boundaries.remove_side(predicted_v, Orientation.all)
        p_prime = Boundaries.remove_side(p_prime, Orientation.all)
        self.__fields_matrix[Field.p] = self.__fields_matrix[Field.p] + p_prime
        gradient_x_p_prime = self.__gradient_x_p_prime_for_u_field(p_prime)
        gradient_x_p_prime = np.multiply(gradient_x_p_prime, self.__delta.t)
        self.__fields_matrix[Field.u] = predicted_u - gradient_x_p_prime
        gradient_y_p_prime = self.__gradient_y_p_prime_for_v_field(p_prime)
        gradient_y_p_prime = np.multiply(gradient_y_p_prime, self.__delta.t)
        self.__fields_matrix[Field.v] = predicted_v - gradient_y_p_prime

    def __derivative_t_for_momentum_conservation_u(self):
        divergence_t = self.__fields_matrix[Field.u]
        divergence_t = np.multiply(divergence_t, -1 / self.__delta.t)
        divergence_t = np.multiply(divergence_t.T, self.__delta.y).T
        divergence_t = np.multiply(divergence_t, self.__delta.half_x_no_boundaries)
        return divergence_t

    def __derivative_t_for_momentum_conservation_v(self):
        divergence_v_delta_t = self.__fields_matrix[Field.v]
        divergence_v_delta_t = np.multiply(divergence_v_delta_t, -1 / self.__delta.t)
        divergence_v_delta_t = np.multiply(divergence_v_delta_t.T, self.__delta.half_y_no_boundaries).T
        divergence_v_delta_t = np.multiply(divergence_v_delta_t, self.__delta.x)
        return divergence_v_delta_t

    def __non_linear_parameters_x(self):
        # P = plus  M = minus h = half
        u_matrix = self.__fields_matrix[Field.u]
        v_matrix = self.__fields_matrix[Field.v]

        # find the matrices without the unneeded columns and rows
        u_iP1_j = Boundaries.remove_side(u_matrix, Orientation.left)
        u_iP1_j = Boundaries.add_boundaries(u_iP1_j, self.__boundaries, Field.u, Orientation.right)
        u_i_j = u_matrix
        u_iM1_j = Boundaries.remove_side(u_matrix, Orientation.right)
        u_iM1_j = Boundaries.add_boundaries(u_iM1_j, self.__boundaries, Field.u, Orientation.left)
        u_i_jP1 = Boundaries.remove_side(u_matrix, Orientation.bottom)
        u_i_jM1 = Boundaries.remove_side(u_matrix, Orientation.top)

        v_i_j = Boundaries.add_boundaries(v_matrix, self.__boundaries, Field.v, Orientation.top)
        v_i_j = Boundaries.remove_side(v_i_j, Orientation.left)
        v_iM1_j = Boundaries.add_boundaries(v_matrix, self.__boundaries, Field.v, Orientation.top)
        v_iM1_j = Boundaries.remove_side(v_iM1_j, Orientation.right)
        v_i_jM1 = Boundaries.add_boundaries(v_matrix, self.__boundaries, Field.v, Orientation.bottom)
        v_i_jM1 = Boundaries.remove_side(v_i_jM1, Orientation.left)
        v_iM1_jM1 = Boundaries.add_boundaries(v_matrix, self.__boundaries, Field.v, Orientation.bottom)
        v_iM1_jM1 = Boundaries.remove_side(v_iM1_jM1, Orientation.right)

        # find matrices with half indexes
        u_iPh_j = (u_iP1_j + u_i_j) / 2
        u_iMh_j = (u_iM1_j + u_i_j) / 2
        u_i_jh = (u_i_jP1 + u_i_jM1) / 2
        u_i_jPh = Boundaries.add_boundaries(u_i_jh, self.__boundaries, Field.u, Orientation.top)
        u_i_jMh = Boundaries.add_boundaries(u_i_jh, self.__boundaries, Field.u, Orientation.bottom)

        v_iMh_jP1 = (v_i_j + v_iM1_j) / 2
        v_iMh_jM1 = (v_i_jM1 + v_iM1_jM1) / 2

        # final expression
        delta_y_multiplication = np.multiply(u_iPh_j, u_iPh_j)
        delta_y_multiplication -= np.multiply(u_iMh_j, u_iMh_j)
        delta_x_multiplication = np.multiply(u_i_jPh, v_iMh_jP1)
        delta_x_multiplication -= np.multiply(u_i_jMh, v_iMh_jM1)
        non_linear_parameters_x = np.multiply(delta_y_multiplication.T, self.__delta.y).T + \
                                  np.multiply(delta_x_multiplication, self.__delta.half_x_no_boundaries)

        return non_linear_parameters_x

    def __non_linear_parameters_y(self):
        # P = plus  M = minus h = half
        u_matrix = self.__fields_matrix[Field.u]
        v_matrix = self.__fields_matrix[Field.v]

        # find the matrices without the unneeded columns and rows
        u_iP1_jM1 = Boundaries.add_boundaries(u_matrix, self.__boundaries, Field.u, Orientation.right)
        u_iP1_jM1 = Boundaries.remove_side(u_iP1_jM1, Orientation.top)
        u_iP1_j = Boundaries.add_boundaries(u_matrix, self.__boundaries, Field.u, Orientation.right)
        u_iP1_j = Boundaries.remove_side(u_iP1_j, Orientation.bottom)
        u_i_jM1 = Boundaries.add_boundaries(u_matrix, self.__boundaries, Field.u, Orientation.left)
        u_i_jM1 = Boundaries.remove_side(u_i_jM1, Orientation.top)
        u_i_j = Boundaries.add_boundaries(u_matrix, self.__boundaries, Field.u, Orientation.left)
        u_i_j = Boundaries.remove_side(u_i_j, Orientation.bottom)

        v_i_jP1 = Boundaries.remove_side(v_matrix, Orientation.bottom)
        v_i_jP1 = Boundaries.add_boundaries(v_i_jP1, self.__boundaries, Field.v, Orientation.top)
        v_i_j = v_matrix
        v_i_jM1 = Boundaries.remove_side(v_matrix, Orientation.top)
        v_i_jM1 = Boundaries.add_boundaries(v_i_jM1, self.__boundaries, Field.v, Orientation.bottom)
        v_iP1_j = Boundaries.remove_side(v_matrix, Orientation.left)
        v_iM1_j = Boundaries.remove_side(v_matrix, Orientation.right)

        # find matrices with half indexes
        u_iP1_jMh = (u_iP1_jM1 + u_iP1_j) / 2
        u_i_jMh = (u_i_jM1 + u_i_j) / 2

        v_i_jPh = (v_i_j + v_i_jP1) / 2
        v_i_jMh = (v_i_j + v_i_jM1) / 2
        v_middle_i_j = (v_iP1_j + v_iM1_j) / 2
        v_iPh_j = Boundaries.add_boundaries(v_middle_i_j, self.__boundaries, Field.v, Orientation.right)
        v_iMh_j = Boundaries.add_boundaries(v_middle_i_j, self.__boundaries, Field.v, Orientation.left)

        # final expression
        delta_x_multiplication = np.multiply(v_i_jPh, v_i_jPh)
        delta_x_multiplication -= np.multiply(v_i_jMh, v_i_jMh)
        delta_y_multiplication = np.multiply(u_iP1_jMh, v_iPh_j)
        delta_y_multiplication -= np.multiply(u_i_jMh, v_iMh_j)
        non_linear_parameters_y = np.multiply(delta_y_multiplication.T, self.__delta.half_y_no_boundaries).T + \
                                  np.multiply(delta_x_multiplication, self.__delta.x)

        return non_linear_parameters_y

    def __gradient_p_predicted_u(self):
        # P = plus
        p_iP1_j = Boundaries.remove_side(self.__fields_matrix[Field.p], Orientation.left)
        p_i_j = Boundaries.remove_side(self.__fields_matrix[Field.p], Orientation.right)
        results = p_iP1_j - p_i_j
        results = np.multiply(results.T, self.__delta.y).T
        return results

    def __gradient_p_predicted_v(self):
        # P = plus
        p_i_jP1 = Boundaries.remove_side(self.__fields_matrix[Field.p], Orientation.bottom)
        p_i_j = Boundaries.remove_side(self.__fields_matrix[Field.p], Orientation.top)
        results = p_i_jP1 - p_i_j
        results = np.multiply(results, self.__delta.x)
        return results

    def __gradient_x_p_prime_for_u_field(self, p_prime):
        # P = plus
        p_prime_iP1_j = Boundaries.remove_side(p_prime, Orientation.left)
        p_prime_i_j = Boundaries.remove_side(p_prime, Orientation.right)
        results = (p_prime_iP1_j - p_prime_i_j) / self.__delta.half_x_no_boundaries
        return results

    def __divergence_x_field_u(self, predicted_u):
        # P = plus
        predicted_u_iP1_j = Boundaries.add_boundaries(predicted_u, self.__boundaries, Field.u, Orientation.right)
        predicted_u_i_j = Boundaries.add_boundaries(predicted_u, self.__boundaries, Field.u, Orientation.left)
        results = (predicted_u_iP1_j - predicted_u_i_j) / self.__delta.x
        return results

    def __gradient_y_p_prime_for_v_field(self, p_prime):
        # P = plus
        p_prime_i_jP1 = Boundaries.remove_side(p_prime, Orientation.bottom)
        p_prime_i_j = Boundaries.remove_side(p_prime, Orientation.top)
        results = ((p_prime_i_jP1 - p_prime_i_j).T / self.__delta.half_y_no_boundaries).T
        return results

    def __divergence_y_field_v(self, predicted_v):
        # M = minus
        predicted_v_i_jP1 = Boundaries.add_boundaries(predicted_v, self.__boundaries, Field.v, Orientation.top)
        predicted_v_i_j = Boundaries.add_boundaries(predicted_v, self.__boundaries, Field.v, Orientation.bottom)
        results = ((predicted_v_i_jP1 - predicted_v_i_j).T / self.__delta.y).T
        return results

    ###################################### On Index Fields

    def __get_index_u_matrix(self):
        left_boundary = np.array([self.__boundaries.get_boundary(Orientation.left, Field.u)]).T
        right_boundary = np.array([self.__boundaries.get_boundary(Orientation.right, Field.u)]).T
        index_u_matrix = np.concatenate((left_boundary, self.__fields_matrix[Field.u], right_boundary), axis=1)
        index_u_matrix = (Boundaries.remove_side(index_u_matrix, Orientation.top) +
                          Boundaries.remove_side(index_u_matrix, Orientation.bottom)) / 2
        index_u_matrix = Boundaries.add_boundaries(index_u_matrix, self.__boundaries, Field.u,
                                                   Orientation.bottom, with_side_boundaries=True)
        index_u_matrix = Boundaries.add_boundaries(index_u_matrix, self.__boundaries, Field.u,
                                                   Orientation.top, with_side_boundaries=True)
        return index_u_matrix

    def __get_index_v_matrix(self):
        bottom_boundary = [self.__boundaries.get_boundary(Orientation.bottom, Field.v)]
        top_boundary = [self.__boundaries.get_boundary(Orientation.top, Field.v)]
        index_v_matrix = np.concatenate((bottom_boundary, self.__fields_matrix[Field.v], top_boundary), axis=0)
        index_v_matrix = (Boundaries.remove_side(index_v_matrix, Orientation.left) +
                          Boundaries.remove_side(index_v_matrix, Orientation.right)) / 2
        index_v_matrix = Boundaries.add_boundaries(index_v_matrix, self.__boundaries, Field.v,
                                                   Orientation.left, with_side_boundaries=True)
        index_v_matrix = Boundaries.add_boundaries(index_v_matrix, self.__boundaries, Field.v,
                                                   Orientation.right, with_side_boundaries=True)
        return index_v_matrix

    ###################################### Data options

    def quiver(self):
        x = np.append([0], self.__delta.x).cumsum()
        y = np.append([0], self.__delta.y).cumsum()
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