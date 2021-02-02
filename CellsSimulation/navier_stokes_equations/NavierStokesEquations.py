import numpy as np
from LaplaceOperator import LaplaceOperator
import Constants as C
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import GeneralEnums as E
from Boundaries import Boundaries


class NavierStokesEquations:

    def __init__(self, fields, grid, boundaries, delta_t,
                 boundary_conditions_type=E.BoundaryConditionsType.neumann,
                 information=E.Information.none):

        self.__fields = fields
        self.__grid = grid
        self.__boundaries = boundaries
        self.__delta_t = delta_t
        self.__information = information
        # decide the pressure boundaries type: Dirichlet or Neumann
        self.__boundary_conditions_type = boundary_conditions_type

        # make the laplacian operators
        self.__create_p_prime_laplacian()
        self.__create_predicted_u_laplacian()
        self.__create_predicted_v_laplacian()

    ###################################### Create laplacians

    def __create_p_prime_laplacian(self):
        self.__laplace_operator_p_prime = LaplaceOperator(self.__grid, E.Field.p,
                                                          E.BoundaryConditionsType.neumann)

    def __create_predicted_u_laplacian(self):
        self.__laplace_operator_predicted_u = LaplaceOperator(self.__grid, E.Field.u,
                                                              E.BoundaryConditionsType.dirichlet)

        self.__laplace_operator_predicted_u.multiply_operators_matrix(1 / C.RE)

        # add delta_x*delta_y/delta_t
        u_matrix_side_length = (len(self.__grid.delta_x) + 1) * (len(self.__grid.delta_half_y) + 1)
        u_identity_matrix = np.full(u_matrix_side_length, 1)

        delta_x = Boundaries.remove_side(self.__grid.delta_half_x, E.Orientation.all)
        delta_x = np.concatenate(([0], delta_x, [0]))
        delta_x = np.tile(delta_x, (len(self.__grid.delta_y) + 2))
        delta_y = np.concatenate(([0], self.__grid.delta_y, [0]))
        delta_y = np.repeat(delta_y, len(self.__grid.delta_half_x))

        u_identity_matrix = np.multiply(u_identity_matrix, delta_x)
        u_identity_matrix = np.multiply(u_identity_matrix.T, delta_y).T
        u_identity_matrix = np.multiply(u_identity_matrix, 1 / -self.__delta_t)
        u_identity_matrix = sparse.spdiags(u_identity_matrix, [0],
                                           u_matrix_side_length, u_matrix_side_length, format='csr')

        self.__laplace_operator_predicted_u.add_matrix_to_operators_matrix(u_identity_matrix)

    def __create_predicted_v_laplacian(self):
        self.__laplace_operator_predicted_v = LaplaceOperator(self.__grid, E.Field.v,
                                                              E.BoundaryConditionsType.dirichlet)

        self.__laplace_operator_predicted_v.multiply_operators_matrix(1 / C.RE)

        # add delta_x*delta_y/delta_t
        v_matrix_side_length = (len(self.__grid.delta_x) + 1) * (len(self.__grid.delta_half_y) + 1)
        v_identity_matrix = np.full(v_matrix_side_length, 1)

        delta_x = np.concatenate(([0], self.__grid.delta_x, [0]))
        delta_x = np.tile(delta_x, len(self.__grid.delta_half_x))
        delta_y = Boundaries.remove_side(self.__grid.delta_half_y, E.Orientation.all)
        delta_y = np.concatenate(([0], delta_y, [0]))
        delta_y = np.repeat(delta_y, (len(self.__grid.delta_x) + 2))

        v_identity_matrix = np.multiply(v_identity_matrix, delta_x)
        v_identity_matrix = np.multiply(v_identity_matrix, delta_y)
        v_identity_matrix = np.multiply(v_identity_matrix, 1 / -self.__delta_t)
        v_identity_matrix = sparse.spdiags(v_identity_matrix, [0],
                                           v_matrix_side_length, v_matrix_side_length, format='csr')

        self.__laplace_operator_predicted_v.add_matrix_to_operators_matrix(v_identity_matrix)

    def next_step(self):
        predicted_u = self.__calculate_predicted_u()
        predicted_v = self.__calculate_predicted_v()
        p_prime = self.__calculate_p_prime(predicted_u, predicted_v)
        self.__calculate_new_fields(p_prime, predicted_u, predicted_v)
        if self.__information & E.Information.check_divergent:
            self.__check_divergence()
        if self.__information & E.Information.check_gradient_p_dot_u_vector:
            self.__check_gradient_p_dot_u_vector()
        if self.__information & E.Information.check_num_3:
            self.__check_num_3()

    def __check_divergence(self):
        u_divergence_x = self.__divergence_x_field_u(self.__fields.u)
        v_divergence_y = self.__divergence_y_field_v(self.__fields.v)
        matrix = u_divergence_x + v_divergence_y
        print("Max divergence = ", matrix.max())

    def __check_gradient_p_dot_u_vector(self):
        # P = plus  M = minus h = half
        p_i_j = self.__fields.p
        p_iP1_j = Boundaries.remove_side(p_i_j, E.Orientation.left)
        p_iM1_j = Boundaries.remove_side(p_i_j, E.Orientation.right)
        p_i_jP1 = Boundaries.remove_side(p_i_j, E.Orientation.bottom)
        p_i_jM1 = Boundaries.remove_side(p_i_j, E.Orientation.top)

        p_middle_i_j = (p_iP1_j + p_iM1_j) / 2
        p_iPh_j = Boundaries.add_boundaries(p_middle_i_j, self.__boundaries, E.Field.p, E.Orientation.right)
        p_iMh_j = Boundaries.add_boundaries(p_middle_i_j, self.__boundaries, E.Field.p, E.Orientation.left)
        p_i_middle_j = (p_i_jP1 + p_i_jM1) / 2
        p_i_jPh = Boundaries.add_boundaries(p_i_middle_j, self.__boundaries, E.Field.p, E.Orientation.top)
        p_i_jMh = Boundaries.add_boundaries(p_i_middle_j, self.__boundaries, E.Field.p, E.Orientation.bottom)

        u_i_j = self.__fields.u
        u_iM1_j = Boundaries.add_boundaries(u_i_j, self.__boundaries, E.Field.u, E.Orientation.left)
        u_i_j = Boundaries.add_boundaries(u_i_j, self.__boundaries, E.Field.u, E.Orientation.right)
        v_i_j = self.__fields.v
        v_i_jM1 = Boundaries.add_boundaries(v_i_j, self.__boundaries, E.Field.v, E.Orientation.bottom)
        v_i_j = Boundaries.add_boundaries(v_i_j, self.__boundaries, E.Field.v, E.Orientation.top)

        results = np.multiply(p_iPh_j, u_i_j)
        results -= np.multiply(p_iMh_j, u_iM1_j)
        results += np.multiply(p_i_jPh, v_i_j)
        results -= np.multiply(p_i_jMh, v_i_jM1)

        print("Max gradient p dot u vector = ", results.max())

    def __check_num_3(self):
        # P = plus  M = minus h = half

        # set v and u on p grid and both derivatives x on u and y on v
        u_i_j = Boundaries.add_boundaries(self.__fields.u,
                                          self.__boundaries, E.Field.u, E.Orientation.left)
        u_iP1_j = Boundaries.add_boundaries(self.__fields.u,
                                            self.__boundaries, E.Field.u, E.Orientation.right)
        v_i_j = Boundaries.add_boundaries(self.__fields.v,
                                          self.__boundaries, E.Field.v, E.Orientation.bottom)
        v_i_jP1 = Boundaries.add_boundaries(self.__fields.v,
                                            self.__boundaries, E.Field.v, E.Orientation.top)

        u_on_p_grid = (u_i_j + u_iP1_j) / 2
        v_on_p_grid = (v_i_j + v_i_jP1) / 2
        derivative_u_of_x = np.multiply(u_iP1_j - u_i_j, 1 / self.__grid.delta_x)
        derivative_v_of_y = np.multiply((v_i_jP1 - v_i_j).T, 1 / self.__grid.delta_y).T

        # set derivatives u of y and v of x
        u_i_j = Boundaries.add_boundaries(u_i_j, self.__boundaries, E.Field.u, E.Orientation.right)
        u_i_jP1 = Boundaries.remove_side(u_i_j, E.Orientation.bottom)
        u_i_j = Boundaries.remove_side(u_i_j, E.Orientation.top)
        v_i_j = Boundaries.add_boundaries(v_i_j, self.__boundaries, E.Field.v, E.Orientation.top)
        v_iP1_j = Boundaries.remove_side(v_i_j, E.Orientation.left)
        v_i_j = Boundaries.remove_side(v_i_j, E.Orientation.right)

        derivative_u_of_y = np.multiply((u_i_jP1 - u_i_j).T, 1 / self.__grid.delta_half_y_no_boundaries).T
        derivative_u_of_y = Boundaries.add_boundaries(derivative_u_of_y, self.__boundaries, E.Field.u, E.Orientation.top,
                                                      with_edge_boundaries=True)
        derivative_u_of_y = Boundaries.add_boundaries(derivative_u_of_y, self.__boundaries, E.Field.u, E.Orientation.bottom,
                                                      with_edge_boundaries=True)
        # find derivative_u_of_y on p grid
        derivative_u_of_y_top_left_corner = Boundaries.remove_side(derivative_u_of_y, E.Orientation.bottom)
        derivative_u_of_y_top_left_corner = Boundaries.remove_side(derivative_u_of_y_top_left_corner,
                                                                   E.Orientation.right)
        derivative_u_of_y_top_right_corner = Boundaries.remove_side(derivative_u_of_y, E.Orientation.bottom)
        derivative_u_of_y_top_right_corner = Boundaries.remove_side(derivative_u_of_y_top_right_corner,
                                                                    E.Orientation.left)
        derivative_u_of_y_bottom_right_corner = Boundaries.remove_side(derivative_u_of_y, E.Orientation.top)
        derivative_u_of_y_bottom_right_corner = Boundaries.remove_side(derivative_u_of_y_bottom_right_corner,
                                                                       E.Orientation.left)
        derivative_u_of_y_bottom_left_corner = Boundaries.remove_side(derivative_u_of_y, E.Orientation.top)
        derivative_u_of_y_bottom_left_corner = Boundaries.remove_side(derivative_u_of_y_bottom_left_corner,
                                                                      E.Orientation.right)
        derivative_u_of_y = (derivative_u_of_y_top_left_corner
                             + derivative_u_of_y_top_right_corner
                             + derivative_u_of_y_bottom_right_corner
                             + derivative_u_of_y_bottom_left_corner) / 4

        derivative_v_of_x = np.multiply(v_iP1_j - v_i_j, 1 / self.__grid.delta_half_x_no_boundaries)
        derivative_v_of_x = Boundaries.add_boundaries(derivative_v_of_x, self.__boundaries, E.Field.v, E.Orientation.left,
                                                      with_edge_boundaries=True)
        derivative_v_of_x = Boundaries.add_boundaries(derivative_v_of_x, self.__boundaries, E.Field.v, E.Orientation.right,
                                                      with_edge_boundaries=True)

        # find derivative_v_of_x on p grid
        derivative_v_of_x_top_left_corner = Boundaries.remove_side(derivative_v_of_x, E.Orientation.bottom)
        derivative_v_of_x_top_left_corner = Boundaries.remove_side(derivative_v_of_x_top_left_corner,
                                                                   E.Orientation.right)
        derivative_v_of_x_top_right_corner = Boundaries.remove_side(derivative_v_of_x, E.Orientation.bottom)
        derivative_v_of_x_top_right_corner = Boundaries.remove_side(derivative_v_of_x_top_right_corner,
                                                                    E.Orientation.left)
        derivative_v_of_x_bottom_right_corner = Boundaries.remove_side(derivative_v_of_x, E.Orientation.top)
        derivative_v_of_x_bottom_right_corner = Boundaries.remove_side(derivative_v_of_x_bottom_right_corner,
                                                                       E.Orientation.left)
        derivative_v_of_x_bottom_left_corner = Boundaries.remove_side(derivative_v_of_x, E.Orientation.top)
        derivative_v_of_x_bottom_left_corner = Boundaries.remove_side(derivative_v_of_x_bottom_left_corner,
                                                                      E.Orientation.right)
        derivative_v_of_x = (derivative_v_of_x_top_left_corner
                             + derivative_v_of_x_top_right_corner
                             + derivative_v_of_x_bottom_right_corner
                             + derivative_v_of_x_bottom_left_corner) / 4

        results = np.multiply(np.multiply(u_on_p_grid, u_on_p_grid), derivative_u_of_x) \
                  + np.multiply(np.multiply(u_on_p_grid, v_on_p_grid), derivative_u_of_y + derivative_v_of_x) \
                  + np.multiply(np.multiply(u_on_p_grid, u_on_p_grid), derivative_v_of_y)

        print("Max check number 3 = ", results.max())

    def __calculate_predicted_u(self):
        gradient_p = self.__gradient_p_predicted_u()
        derivative_t = self.__derivative_t_for_momentum_conservation_u()
        non_linear = self.__non_linear_parameters_x()
        right_side_predicted_u = gradient_p - derivative_t + non_linear
        right_side_predicted_u = Boundaries.add_boundaries(right_side_predicted_u, self.__boundaries,
                                                           E.Field.u, E.Orientation.all)
        predicted_u = self.__laplace_operator_predicted_u.solve(right_side_predicted_u)
        return predicted_u

    def __calculate_predicted_v(self):
        gradient_p = self.__gradient_p_predicted_v()
        derivative_t = self.__derivative_t_for_momentum_conservation_v()
        non_linear_parameters = self.__non_linear_parameters_y()
        right_side_predicted_v = gradient_p - derivative_t + non_linear_parameters
        right_side_predicted_v = Boundaries.add_boundaries(right_side_predicted_v, self.__boundaries,
                                                           E.Field.v, E.Orientation.all)
        predicted_v = self.__laplace_operator_predicted_v.solve(right_side_predicted_v)
        return predicted_v

    def __calculate_p_prime(self, predicted_u, predicted_v):
        predicted_u = Boundaries.remove_side(predicted_u, E.Orientation.all)
        predicted_v = Boundaries.remove_side(predicted_v, E.Orientation.all)
        divergence_x_predicted_u = self.__divergence_x_field_u(predicted_u)
        divergence_y_predicted_v = self.__divergence_y_field_v(predicted_v)
        right_side_p_prime = divergence_x_predicted_u + divergence_y_predicted_v
        right_side_p_prime /= self.__delta_t
        right_side_p_prime[0][0] = 0
        right_side_p_prime = Boundaries.add_boundaries(right_side_p_prime, self.__boundaries,
                                                       E.Field.p, E.Orientation.all)
        p_prime = self.__laplace_operator_p_prime.solve(right_side_p_prime)
        return p_prime

    def __calculate_new_fields(self, p_prime, predicted_u, predicted_v):
        predicted_u = Boundaries.remove_side(predicted_u, E.Orientation.all)
        predicted_v = Boundaries.remove_side(predicted_v, E.Orientation.all)
        p_prime = Boundaries.remove_side(p_prime, E.Orientation.all)
        self.__fields.p = self.__fields.p + p_prime
        gradient_x_p_prime = self.__gradient_x_p_prime_for_u_field(p_prime)
        gradient_x_p_prime = np.multiply(gradient_x_p_prime, self.__delta_t)
        self.__fields.u = predicted_u - gradient_x_p_prime
        gradient_y_p_prime = self.__gradient_y_p_prime_for_v_field(p_prime)
        gradient_y_p_prime = np.multiply(gradient_y_p_prime, self.__delta_t)
        self.__fields.v = predicted_v - gradient_y_p_prime

    def __derivative_t_for_momentum_conservation_u(self):
        divergence_t = self.__fields.u
        divergence_t = np.multiply(divergence_t, 1 / self.__delta_t)
        divergence_t = np.multiply(divergence_t.T, self.__grid.delta_y).T
        divergence_t = np.multiply(divergence_t, self.__grid.delta_half_x_no_boundaries)
        return divergence_t

    def __derivative_t_for_momentum_conservation_v(self):
        divergence_v_delta_t = self.__fields.v
        divergence_v_delta_t = np.multiply(divergence_v_delta_t, 1 / self.__delta_t)
        divergence_v_delta_t = np.multiply(divergence_v_delta_t.T, self.__grid.delta_half_y_no_boundaries).T
        divergence_v_delta_t = np.multiply(divergence_v_delta_t, self.__grid.delta_x)
        return divergence_v_delta_t

    def __non_linear_parameters_x(self):
        # P = plus  M = minus h = half
        u_matrix = self.__fields.u
        v_matrix = self.__fields.v

        # find the matrices without the unneeded columns and rows
        u_iP1_j = Boundaries.remove_side(u_matrix, E.Orientation.left)
        u_iP1_j = Boundaries.add_boundaries(u_iP1_j, self.__boundaries, E.Field.u, E.Orientation.right)
        u_i_j = u_matrix
        u_iM1_j = Boundaries.remove_side(u_matrix, E.Orientation.right)
        u_iM1_j = Boundaries.add_boundaries(u_iM1_j, self.__boundaries, E.Field.u, E.Orientation.left)
        u_i_jP1 = Boundaries.remove_side(u_matrix, E.Orientation.bottom)
        u_i_jM1 = Boundaries.remove_side(u_matrix, E.Orientation.top)

        v_i_j = Boundaries.add_boundaries(v_matrix, self.__boundaries, E.Field.v, E.Orientation.top)
        v_i_j = Boundaries.remove_side(v_i_j, E.Orientation.left)
        v_iM1_j = Boundaries.add_boundaries(v_matrix, self.__boundaries, E.Field.v, E.Orientation.top)
        v_iM1_j = Boundaries.remove_side(v_iM1_j, E.Orientation.right)
        v_i_jM1 = Boundaries.add_boundaries(v_matrix, self.__boundaries, E.Field.v, E.Orientation.bottom)
        v_i_jM1 = Boundaries.remove_side(v_i_jM1, E.Orientation.left)
        v_iM1_jM1 = Boundaries.add_boundaries(v_matrix, self.__boundaries, E.Field.v, E.Orientation.bottom)
        v_iM1_jM1 = Boundaries.remove_side(v_iM1_jM1, E.Orientation.right)

        # find matrices with half indexes
        u_iPh_j = (u_iP1_j + u_i_j) / 2
        u_iMh_j = (u_iM1_j + u_i_j) / 2
        u_i_jh = (u_i_jP1 + u_i_jM1) / 2
        u_i_jPh = Boundaries.add_boundaries(u_i_jh, self.__boundaries, E.Field.u, E.Orientation.top)
        u_i_jMh = Boundaries.add_boundaries(u_i_jh, self.__boundaries, E.Field.u, E.Orientation.bottom)

        v_iMh_jP1 = (v_i_j + v_iM1_j) / 2
        v_iMh_jM1 = (v_i_jM1 + v_iM1_jM1) / 2

        # final expression
        delta_y_multiplication = np.multiply(u_iPh_j, u_iPh_j)
        delta_y_multiplication -= np.multiply(u_iMh_j, u_iMh_j)
        delta_x_multiplication = np.multiply(u_i_jPh, v_iMh_jP1)
        delta_x_multiplication -= np.multiply(u_i_jMh, v_iMh_jM1)
        non_linear_parameters_x = np.multiply(delta_y_multiplication.T, self.__grid.delta_y).T + \
                                  np.multiply(delta_x_multiplication, self.__grid.delta_half_x_no_boundaries)

        return non_linear_parameters_x

    def __non_linear_parameters_y(self):
        # P = plus  M = minus h = half
        u_matrix = self.__fields.u
        v_matrix = self.__fields.v

        # find the matrices without the unneeded columns and rows
        u_iP1_jM1 = Boundaries.add_boundaries(u_matrix, self.__boundaries, E.Field.u, E.Orientation.right)
        u_iP1_jM1 = Boundaries.remove_side(u_iP1_jM1, E.Orientation.top)
        u_iP1_j = Boundaries.add_boundaries(u_matrix, self.__boundaries, E.Field.u, E.Orientation.right)
        u_iP1_j = Boundaries.remove_side(u_iP1_j, E.Orientation.bottom)
        u_i_jM1 = Boundaries.add_boundaries(u_matrix, self.__boundaries, E.Field.u, E.Orientation.left)
        u_i_jM1 = Boundaries.remove_side(u_i_jM1, E.Orientation.top)
        u_i_j = Boundaries.add_boundaries(u_matrix, self.__boundaries, E.Field.u, E.Orientation.left)
        u_i_j = Boundaries.remove_side(u_i_j, E.Orientation.bottom)

        v_i_jP1 = Boundaries.remove_side(v_matrix, E.Orientation.bottom)
        v_i_jP1 = Boundaries.add_boundaries(v_i_jP1, self.__boundaries, E.Field.v, E.Orientation.top)
        v_i_j = v_matrix
        v_i_jM1 = Boundaries.remove_side(v_matrix, E.Orientation.top)
        v_i_jM1 = Boundaries.add_boundaries(v_i_jM1, self.__boundaries, E.Field.v, E.Orientation.bottom)
        v_iP1_j = Boundaries.remove_side(v_matrix, E.Orientation.left)
        v_iM1_j = Boundaries.remove_side(v_matrix, E.Orientation.right)

        # find matrices with half indexes
        u_iP1_jMh = (u_iP1_jM1 + u_iP1_j) / 2
        u_i_jMh = (u_i_jM1 + u_i_j) / 2

        v_i_jPh = (v_i_j + v_i_jP1) / 2
        v_i_jMh = (v_i_j + v_i_jM1) / 2
        v_middle_i_j = (v_iP1_j + v_iM1_j) / 2
        v_iPh_j = Boundaries.add_boundaries(v_middle_i_j, self.__boundaries, E.Field.v, E.Orientation.right)
        v_iMh_j = Boundaries.add_boundaries(v_middle_i_j, self.__boundaries, E.Field.v, E.Orientation.left)

        # final expression
        delta_x_multiplication = np.multiply(v_i_jPh, v_i_jPh)
        delta_x_multiplication -= np.multiply(v_i_jMh, v_i_jMh)
        delta_y_multiplication = np.multiply(u_iP1_jMh, v_iPh_j)
        delta_y_multiplication -= np.multiply(u_i_jMh, v_iMh_j)
        non_linear_parameters_y = np.multiply(delta_y_multiplication.T, self.__grid.delta_half_y_no_boundaries).T + \
                                  np.multiply(delta_x_multiplication, self.__grid.delta_x)

        return non_linear_parameters_y

    def __gradient_p_predicted_u(self):
        # P = plus
        p_iP1_j = Boundaries.remove_side(self.__fields.p, E.Orientation.left)
        p_i_j = Boundaries.remove_side(self.__fields.p, E.Orientation.right)
        results = p_iP1_j - p_i_j
        results = np.multiply(results.T, self.__grid.delta_y).T
        return results

    def __gradient_p_predicted_v(self):
        # P = plus
        p_i_jP1 = Boundaries.remove_side(self.__fields.p, E.Orientation.bottom)
        p_i_j = Boundaries.remove_side(self.__fields.p, E.Orientation.top)
        results = p_i_jP1 - p_i_j
        results = np.multiply(results, self.__grid.delta_x)
        return results

    def __gradient_x_p_prime_for_u_field(self, p_prime):
        # P = plus
        p_prime_iP1_j = Boundaries.remove_side(p_prime, E.Orientation.left)
        p_prime_i_j = Boundaries.remove_side(p_prime, E.Orientation.right)
        results = (p_prime_iP1_j - p_prime_i_j) / self.__grid.delta_half_x_no_boundaries
        return results

    def __divergence_x_field_u(self, predicted_u):
        # P = plus
        predicted_u_iP1_j = Boundaries.add_boundaries(predicted_u, self.__boundaries, E.Field.u, E.Orientation.right)
        predicted_u_i_j = Boundaries.add_boundaries(predicted_u, self.__boundaries, E.Field.u, E.Orientation.left)
        results = (predicted_u_iP1_j - predicted_u_i_j) / self.__grid.delta_x
        return results

    def __gradient_y_p_prime_for_v_field(self, p_prime):
        # P = plus
        p_prime_i_jP1 = Boundaries.remove_side(p_prime, E.Orientation.bottom)
        p_prime_i_j = Boundaries.remove_side(p_prime, E.Orientation.top)
        results = ((p_prime_i_jP1 - p_prime_i_j).T / self.__grid.delta_half_y_no_boundaries).T
        return results

    def __divergence_y_field_v(self, predicted_v):
        # M = minus
        predicted_v_i_jP1 = Boundaries.add_boundaries(predicted_v, self.__boundaries, E.Field.v, E.Orientation.top)
        predicted_v_i_j = Boundaries.add_boundaries(predicted_v, self.__boundaries, E.Field.v, E.Orientation.bottom)
        results = ((predicted_v_i_jP1 - predicted_v_i_j).T / self.__grid.delta_y).T
        return results

    ###################################### On Index Fields

    def __get_index_u_matrix(self):
        left_boundary = np.array([self.__boundaries.get_boundary(E.Orientation.left, E.Field.u)]).T
        right_boundary = np.array([self.__boundaries.get_boundary(E.Orientation.right, E.Field.u)]).T
        index_u_matrix = np.concatenate((left_boundary, self.__fields.u, right_boundary), axis=1)
        index_u_matrix = (Boundaries.remove_side(index_u_matrix, E.Orientation.top) +
                          Boundaries.remove_side(index_u_matrix, E.Orientation.bottom)) / 2
        index_u_matrix = Boundaries.add_boundaries(index_u_matrix, self.__boundaries, E.Field.u,
                                                   E.Orientation.bottom, with_edge_boundaries=True)
        index_u_matrix = Boundaries.add_boundaries(index_u_matrix, self.__boundaries, E.Field.u,
                                                   E.Orientation.top, with_edge_boundaries=True)
        return index_u_matrix

    def __get_index_v_matrix(self):
        bottom_boundary = [self.__boundaries.get_boundary(E.Orientation.bottom, E.Field.v)]
        top_boundary = [self.__boundaries.get_boundary(E.Orientation.top, E.Field.v)]
        index_v_matrix = np.concatenate((bottom_boundary, self.__fields.v, top_boundary), axis=0)
        index_v_matrix = (Boundaries.remove_side(index_v_matrix, E.Orientation.left) +
                          Boundaries.remove_side(index_v_matrix, E.Orientation.right)) / 2
        index_v_matrix = Boundaries.add_boundaries(index_v_matrix, self.__boundaries, E.Field.v,
                                                   E.Orientation.left, with_edge_boundaries=True)
        index_v_matrix = Boundaries.add_boundaries(index_v_matrix, self.__boundaries, E.Field.v,
                                                   E.Orientation.right, with_edge_boundaries=True)
        return index_v_matrix

    ###################################### Data options

    def quiver(self):
        x = np.append([0], self.__grid.delta_x).cumsum()
        y = np.append([0], self.__grid.delta_y).cumsum()
        xx, yy = np.meshgrid(x, y)

        index_u_matrix = self.__get_index_u_matrix()
        index_v_matrix = self.__get_index_v_matrix()
        return plt.quiver(xx, yy, index_u_matrix, index_v_matrix)

    ###################################### Setters and Getters

    @property
    def boundary_conditions_type(self):
        return self.__boundary_conditions_type

    @boundary_conditions_type.setter
    def boundary_conditions_type(self, boundary_conditions_type):
        self.__boundary_conditions_type = boundary_conditions_type

    @property
    def fields(self):
        return self.__fields
