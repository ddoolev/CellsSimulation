import numpy as np
from LaplaceOperator import LaplaceOperator
import Constants as C
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from general_enums import Field, Delta, Orientation, BoundaryConditionsType


class NavierStokesEquations:

    def __init__(self, field_matrix, delta_matrix, boundaries,
                 boundary_conditions_type=BoundaryConditionsType.neumann):

        self.__fields_matrix = field_matrix
        self.__delta_matrix = delta_matrix
        self.__boundaries = boundaries
        self.__delta_t = C.DELTA_T
        # decide the pressure boundaries type: Dirichlet or Neumann
        self.__boundary_conditions_type = boundary_conditions_type

        # make the laplacian operators
        self.__create_p_prime_laplacian()
        self.__create_predicted_u_laplacian()
        self.__create_predicted_v_laplacian()

    ###################################### Create laplacians

    def __create_p_prime_laplacian(self):
        self.__laplace_operator_p_prime = LaplaceOperator(self.__delta_matrix[Delta.x],
                                                          self.__delta_matrix[Delta.y],
                                                          Field.p,
                                                          BoundaryConditionsType.neumann)

    def __create_predicted_u_laplacian(self):
        self.__laplace_operator_predicted_u = LaplaceOperator(self.__delta_matrix[Delta.x],
                                                              self.__delta_matrix[Delta.y],
                                                              Field.u,
                                                              BoundaryConditionsType.dirichlet)

        self.__laplace_operator_predicted_u.multiply_operators_matrix(1 / C.Re)

        # add delta_x*delta_y/delta_t

        delta_x_half_grid = self.__create_half_grid(Delta.x)
        delta_y_half_grid = self.__create_half_grid(Delta.y)

        u_matrix_side_length = (len(self.__delta_matrix[Delta.x]) + 1) * (len(delta_y_half_grid) + 1)
        u_identity_matrix = np.full(u_matrix_side_length, 1)

        delta_x = np.concatenate(([0], delta_x_half_grid[1:-1], [0]))
        delta_x = np.tile(delta_x, (len(delta_y_half_grid) + 1))
        delta_y = np.concatenate(([0], self.__delta_matrix[Delta.y], [0]))
        delta_y = np.tile(delta_y, (len(self.__delta_matrix[Delta.x]) + 1))

        u_identity_matrix = np.multiply(u_identity_matrix, delta_x)
        u_identity_matrix = np.multiply(u_identity_matrix, delta_y)
        u_identity_matrix = np.multiply(u_identity_matrix, 1/-C.DELTA_T)
        u_identity_matrix = sparse.spdiags(u_identity_matrix, [0],
                                           u_matrix_side_length, u_matrix_side_length, format='csr')

        self.__laplace_operator_predicted_u.add_matrix_to_operators_matrix(u_identity_matrix)

    def __create_predicted_v_laplacian(self):
        self.__laplace_operator_predicted_v = LaplaceOperator(self.__delta_matrix[Delta.x],
                                                              self.__delta_matrix[Delta.y],
                                                              Field.v,
                                                              BoundaryConditionsType.dirichlet)

        self.__laplace_operator_predicted_v.multiply_operators_matrix(1 / C.Re)

        # add delta_x*delta_y/delta_t
        delta_x_half_grid = self.__create_half_grid(Delta.x)
        delta_y_half_grid = self.__create_half_grid(Delta.y)

        v_matrix_side_length = (len(self.__delta_matrix[Delta.x]) + 1) * (len(delta_y_half_grid) + 1)
        v_identity_matrix = np.full(v_matrix_side_length, 1)

        delta_x = np.concatenate(([0], self.__delta_matrix[Delta.x], [0]))
        delta_x = np.tile(delta_x, (len(self.__delta_matrix[Delta.y]) + 1))
        delta_y = np.concatenate(([0], delta_y_half_grid[1:-1], [0]))
        delta_y = np.tile(delta_y, (len(delta_x_half_grid) + 1))

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
        self.__check_divergence(self.__fields_matrix[Field.u], self.__fields_matrix[Field.v])

    def __check_divergence(self, u, v):
        u_divergence_x = self.__divergence_x_predicted_u(u)
        v_divergence_y = self.__divergence_y_predicted_v(v)
        matrix = u_divergence_x + v_divergence_y
        print(matrix)
        print("\n\n")

    def __calculate_predicted_u(self):
        pressure_terms = self.__pressure_terms_predicted_u()
        divergence_t = self.__divergence_t_for_momentum_conservation_u()
        non_linear = self.__non_linear_parameters_x()
        right_side_predicted_u = pressure_terms - divergence_t + non_linear
        right_side_predicted_u = self.__add_all_boundaries(right_side_predicted_u, Field.u)
        predicted_u = self.__laplace_operator_predicted_u.solve(right_side_predicted_u)
        return predicted_u

    def __calculate_predicted_v(self):
        pressure_terms = self.__pressure_terms_predicted_v()
        derivative_t = self.__derivative_t_for_momentum_conservation_v()
        non_linear_parameters = self.__non_linear_parameters_y()
        right_side_predicted_v = pressure_terms - derivative_t + non_linear_parameters
        right_side_predicted_v = self.__add_all_boundaries(right_side_predicted_v, Field.v)
        predicted_v = self.__laplace_operator_predicted_v.solve(right_side_predicted_v)
        return predicted_v

    def __calculate_p_prime(self, predicted_u, predicted_v):
        predicted_u = self.__remove_boundaries(predicted_u, Orientation.all)
        predicted_v = self.__remove_boundaries(predicted_v, Orientation.all)
        divergence_x_predicted_u = self.__divergence_x_predicted_u(predicted_u)
        divergence_y_predicted_v = self.__divergence_y_predicted_v(predicted_v)
        right_side_p_prime = divergence_x_predicted_u + divergence_y_predicted_v
        right_side_p_prime /= self.__delta_t
        right_side_p_prime[0][0] = 0
        right_side_p_prime = self.__add_all_boundaries(right_side_p_prime, Field.p)
        p_prime = self.__laplace_operator_p_prime.solve(right_side_p_prime)
        return p_prime

    def __calculate_new_fields(self, p_prime, predicted_u, predicted_v):
        predicted_u = self.__remove_boundaries(predicted_u, Orientation.all)
        predicted_v = self.__remove_boundaries(predicted_v, Orientation.all)
        p_prime = self.__remove_boundaries(p_prime, Orientation.all)
        self.__fields_matrix[Field.p] = self.__fields_matrix[Field.p] + p_prime
        gradient_x_p_prime = self.__gradient_x_p_prime_for_u_field(p_prime)
        gradient_x_p_prime = np.multiply(gradient_x_p_prime, self.__delta_t)
        self.__fields_matrix[Field.u] = predicted_u - gradient_x_p_prime
        gradient_y_p_prime = self.__gradient_y_p_prime_for_v_field(p_prime)
        gradient_y_p_prime = np.multiply(gradient_y_p_prime, self.__delta_t)
        self.__fields_matrix[Field.v] = predicted_v - gradient_y_p_prime

    def __divergence_t_for_momentum_conservation_u(self):
        x_half_grid = self.__create_half_grid(Delta.x)
        x_half_grid = self.__remove_array_boundaries_all(x_half_grid)
        divergence_t = self.__fields_matrix[Field.u]
        divergence_t = np.multiply(divergence_t, -1/self.__delta_t)
        divergence_t = np.multiply(divergence_t.T, self.__delta_matrix[Delta.y]).T
        divergence_t = np.multiply(divergence_t, x_half_grid)
        return divergence_t

    def __derivative_t_for_momentum_conservation_v(self):
        y_half_grid = self.__create_half_grid(Delta.y)
        y_half_grid = self.__remove_array_boundaries_all(y_half_grid)
        divergence_v_delta_t = self.__fields_matrix[Field.v]
        divergence_v_delta_t = np.multiply(divergence_v_delta_t, -1/self.__delta_t)
        divergence_v_delta_t = np.multiply(divergence_v_delta_t.T, y_half_grid).T
        divergence_v_delta_t = np.multiply(divergence_v_delta_t, self.__delta_matrix[Delta.x])
        return divergence_v_delta_t

    def __non_linear_parameters_x(self):
        # P = plus  M = minus h = half
        u_matrix = self.__fields_matrix[Field.u]
        v_matrix = self.__fields_matrix[Field.v]

        # find the matrices without the unneeded columns and rows
        u_iP1_j = self.__remove_boundaries(u_matrix, Orientation.left)
        u_iP1_j = self.__add_right_boundary(u_iP1_j, Field.u)
        u_i_j = u_matrix
        u_iM1_j = self.__remove_boundaries(u_matrix, Orientation.right)
        u_iM1_j = self.__add_left_boundary(u_iM1_j, Field.u)
        u_i_jP1 = self.__remove_boundaries(u_matrix, Orientation.bottom)
        u_i_jM1 = self.__remove_boundaries(u_matrix, Orientation.top)

        v_i_j = self.__add_top_boundary(v_matrix, Field.v)
        v_i_j = self.__remove_boundaries(v_i_j, Orientation.left)
        v_iM1_j = self.__add_top_boundary(v_matrix, Field.v)
        v_iM1_j = self.__remove_boundaries(v_iM1_j, Orientation.right)
        v_i_jM1 = self.__add_bottom_boundary(v_matrix, Field.v)
        v_i_jM1 = self.__remove_boundaries(v_i_jM1, Orientation.left)
        v_iM1_jM1 = self.__add_bottom_boundary(v_matrix, Field.v)
        v_iM1_jM1 = self.__remove_boundaries(v_iM1_jM1, Orientation.right)

        # find matrices with half indexes
        u_iPh_j = (u_iP1_j + u_i_j) / 2
        u_iMh_j = (u_iM1_j + u_i_j) / 2
        u_i_jh = (u_i_jP1 + u_i_jM1) / 2
        u_i_jPh = self.__add_top_boundary(u_i_jh, Field.u)
        u_i_jMh = self.__add_bottom_boundary(u_i_jh, Field.u)

        v_iMh_jP1 = (v_i_j + v_iM1_j) / 2
        v_iMh_jM1 = (v_i_jM1 + v_iM1_jM1) / 2

        # final expression
        delta_y_multiplication = np.multiply(u_iPh_j, u_iPh_j)
        delta_y_multiplication -= np.multiply(u_iMh_j, u_iMh_j)
        delta_x_multiplication = np.multiply(u_i_jPh, v_iMh_jP1)
        delta_x_multiplication -= np.multiply(u_i_jMh, v_iMh_jM1)
        delta_x = self.__delta_matrix[Delta.x]
        delta_half_x = (self.__remove_array_boundaries_left(delta_x) +
                        self.__remove_array_boundaries_right(delta_x)) / 2
        non_linear_parameters_x = np.multiply(delta_y_multiplication.T, self.__delta_matrix[Delta.y]).T + \
                                  np.multiply(delta_x_multiplication, delta_half_x)

        return non_linear_parameters_x

    def __non_linear_parameters_y(self):
        # P = plus  M = minus h = half
        u_matrix = self.__fields_matrix[Field.u]
        v_matrix = self.__fields_matrix[Field.v]

        # find the matrices without the unneeded columns and rows
        u_iP1_jM1 = self.__add_right_boundary(u_matrix, Field.u)
        u_iP1_jM1 = self.__remove_boundaries(u_iP1_jM1, Orientation.top)
        u_iP1_j = self.__add_right_boundary(u_matrix, Field.u)
        u_iP1_j = self.__remove_boundaries(u_iP1_j, Orientation.bottom)
        u_i_jM1 = self.__add_left_boundary(u_matrix, Field.u)
        u_i_jM1 = self.__remove_boundaries(u_i_jM1, Orientation.top)
        u_i_j = self.__add_left_boundary(u_matrix, Field.u)
        u_i_j = self.__remove_boundaries(u_i_j, Orientation.bottom)

        v_i_jP1 = self.__remove_boundaries(v_matrix, Orientation.bottom)
        v_i_jP1 = self.__add_top_boundary(v_i_jP1, Field.v)
        v_i_j = v_matrix
        v_i_jM1 = self.__remove_boundaries(v_matrix, Orientation.top)
        v_i_jM1 = self.__add_bottom_boundary(v_i_jM1, Field.v)
        v_iP1_j = self.__remove_boundaries(v_matrix, Orientation.left)
        v_iM1_j = self.__remove_boundaries(v_matrix, Orientation.right)

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
        delta_half_y = (self.__remove_boundaries(delta_y, Orientation.left) +
                        self.__remove_boundaries(delta_y, Orientation.right)) / 2
        non_linear_parameters_y = np.multiply(delta_y_multiplication.T, delta_half_y).T + \
                                  np.multiply(delta_x_multiplication, self.__delta_matrix[Delta.x])

        return non_linear_parameters_y

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

    def __gradient_x_p_prime_for_u_field(self, p_prime):
        # P = plus
        p_prime_iP1_j = self.__remove_boundaries(p_prime, Orientation.left)
        p_prime_i_j = self.__remove_boundaries(p_prime, Orientation.right)
        half_grid_x = self.__create_half_grid(Delta.x)[1:-1]
        results = (p_prime_iP1_j - p_prime_i_j) / half_grid_x
        return results

    def __divergence_x_predicted_u(self, predicted_u):
        # P = plus
        predicted_u_iP1_j = self.__add_right_boundary(predicted_u, Field.u)
        predicted_u_i_j = self.__add_left_boundary(predicted_u, Field.u)
        results = (predicted_u_iP1_j - predicted_u_i_j) / self.__delta_matrix[Delta.x]
        return results

    def __gradient_y_p_prime_for_v_field(self, p_prime):
        # P = plus
        p_prime_i_jP1 = p_prime[1:, :]
        p_prime_i_j = p_prime[:-1, :]
        half_grid_y = self.__create_half_grid(Delta.y)[1:-1]
        results = ((p_prime_i_jP1 - p_prime_i_j).T / half_grid_y).T
        return results

    def __divergence_y_predicted_v(self, predicted_v):
        # M = minus
        predicted_v_i_jP1 = self.__add_top_boundary(predicted_v, Field.v)
        predicted_v_i_j = self.__add_bottom_boundary(predicted_v, Field.v)
        results = ((predicted_v_i_jP1 - predicted_v_i_j).T / self.__delta_matrix[Delta.y]).T
        return results

    # TODO: move this function to a more general place and remove duplicate in LaplaceOperator
    def __create_half_grid(self, delta):
        delta_half_grid = (self.__delta_matrix[delta][1:] + self.__delta_matrix[delta][:-1]) / 2
        delta_half_grid = np.concatenate(([self.__delta_matrix[delta][1] / 2],
                                          delta_half_grid,
                                          [self.__delta_matrix[Delta.y][-1] / 2]))
        return delta_half_grid

    ###################################### Boundaries

    def __add_left_boundary(self, matrix, field, with_top_bottom_boundaries=False):
        left_boundary = self.__boundaries.get_boundary(Orientation.left, field)
        if with_top_bottom_boundaries:
            left_boundary = np.concatenate(([0], left_boundary, [0]))

        left_boundary = np.array([left_boundary]).T
        return np.append(left_boundary, matrix, axis=1)

    def __add_right_boundary(self, matrix, field, with_top_bottom_boundaries=False):
        right_boundary = self.__boundaries.get_boundary(Orientation.right, field)
        if with_top_bottom_boundaries:
            right_boundary = np.concatenate(([0], right_boundary, [0]))

        right_boundary = np.array([right_boundary]).T
        return np.append(matrix, right_boundary, axis=1)

    def __add_bottom_boundary(self, matrix, field, with_left_right_boundaries=False):
        bottom_boundary = self.__boundaries.get_boundary(Orientation.bottom, field)
        if with_left_right_boundaries:
            bottom_boundary = np.concatenate(([0], bottom_boundary, [0]))

        return np.append([bottom_boundary], matrix, axis=0)

    def __add_top_boundary(self, matrix, field, with_left_right_boundaries=False):
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

    @staticmethod
    def __remove_boundaries(array, orientation):
        if array.ndim == 1:
            return NavierStokesEquations.__remove_array_boundaries(array, orientation)
        elif array.ndim == 2:
            return NavierStokesEquations.__remove_matrix_boundaries(array, orientation)
        else:
            raise TypeError("Can not handle array with more than 2 dimensions")

    @staticmethod
    def __remove_matrix_boundaries(matrix, orientation):
        remove_boundary = {
            Orientation.top: NavierStokesEquations.__remove_matrix_boundaries_top(matrix),
            Orientation.bottom: NavierStokesEquations.__remove_matrix_boundaries_bottom(matrix),
            Orientation.left: NavierStokesEquations.__remove_matrix_boundaries_left(matrix),
            Orientation.right: NavierStokesEquations.__remove_matrix_boundaries_right(matrix),
            Orientation.all: NavierStokesEquations.__remove_matrix_boundaries_all(matrix)
        }

        return remove_boundary.get(orientation)

    @staticmethod
    def __remove_array_boundaries(array, orientation):
        remove_boundary = {
            Orientation.left: NavierStokesEquations.__remove_array_boundaries_left(array),
            Orientation.right: NavierStokesEquations.__remove_array_boundaries_right(array),
            Orientation.all: NavierStokesEquations.__remove_array_boundaries_all(array)
        }

        return remove_boundary.get(orientation)

    @staticmethod
    def __remove_matrix_boundaries_top(matrix):
        return matrix[:-1, :]

    @staticmethod
    def __remove_matrix_boundaries_bottom(matrix):
        return matrix[1:, :]

    @staticmethod
    def __remove_matrix_boundaries_left(matrix):
        return matrix[:, 1:]

    @staticmethod
    def __remove_matrix_boundaries_right(matrix):
        return matrix[:, :-1]

    @staticmethod
    def __remove_matrix_boundaries_all(matrix):
        return matrix[1:-1, 1:-1]

    @staticmethod
    def __remove_array_boundaries_all(array):
        return array[1:-1]

    @staticmethod
    def __remove_array_boundaries_left(array):
        return array[1:]

    @staticmethod
    def __remove_array_boundaries_right(array):
        return array[:-1]

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
