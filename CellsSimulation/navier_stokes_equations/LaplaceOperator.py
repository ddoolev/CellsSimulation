import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg as sla
from general_enums import Orientation, BoundaryConditionsType
import numbers


class LaplaceOperator:
    __lu_operator: object

    def __init__(self, delta_x, delta_y, boundary_condition_type):
        self.__boundary_condition_type = boundary_condition_type
        self.__lu_decomposition_updated = False
        self.__operators_matrix = \
            self.__create_laplace_operators_matrix(delta_x, delta_y)
        if boundary_condition_type == BoundaryConditionsType.neumann:
            self.__create_corner_dirichlet_for_neumann()


    def __create_laplace_operators_matrix(self, delta_x, delta_y):
        self.__grid_length_x = len(delta_x) + 1
        self.__grid_length_y = len(delta_y) + 1
        num_of_points_in_matrix_side = self.__grid_length_x * self.__grid_length_y
        operators_matrix_diagonals = [[], [], [], [], []]

        boundary_block_top = self.__create_boundary_block(Orientation.top)
        boundary_block_bottom = self.__create_boundary_block(Orientation.bottom)

        operators_matrix_diagonals = \
            np.concatenate((operators_matrix_diagonals, boundary_block_top), axis=1)

        # Iterate on the matrix blocks
        for i in range(self.__grid_length_x,
                       num_of_points_in_matrix_side - self.__grid_length_x,
                       self.__grid_length_x):
            block = self.__create_laplace_operators_matrix_block(delta_x, delta_y, i)
            operators_matrix_diagonals = \
                np.concatenate((operators_matrix_diagonals, block), axis=1)

        operators_matrix_diagonals = \
            np.concatenate((operators_matrix_diagonals, boundary_block_bottom), axis=1)

        offsets = np.array([self.__grid_length_y, 1, 0, -1, -self.__grid_length_y])

        operators_matrix = sparse.spdiags(
            operators_matrix_diagonals, offsets,
            num_of_points_in_matrix_side, num_of_points_in_matrix_side,
            format='csr').T

        return operators_matrix

    def __create_laplace_operators_matrix_vector(self, delta_x, delta_y, i):
        # create the coefficients.
        # P=plus, M=minus
        x_index = i % self.__grid_length_x
        y_index = i // self.__grid_length_x

        coe_i_jM1 = 2 / (delta_y[y_index - 1] * (delta_y[y_index] + delta_y[y_index - 1]))
        coe_iM1_j = 2 / (delta_x[x_index - 1] * (delta_x[x_index] + delta_x[x_index - 1]))
        coe_i_j = -(2 / (delta_x[x_index] * delta_x[x_index - 1]) + 2 / (delta_y[y_index] * delta_y[y_index - 1]))
        coe_iP1_j = 2 / (delta_x[x_index] * (delta_x[x_index] + delta_x[x_index - 1]))
        coe_i_jP1 = 2 / (delta_y[y_index] * (delta_y[y_index] + delta_y[y_index - 1]))

        coe_vector = [[coe_i_jM1], [coe_iM1_j], [coe_i_j], [coe_iP1_j], [coe_i_jP1]]

        return coe_vector

    def __create_laplace_operators_matrix_block(self, delta_x, delta_y, i):
        block = [[], [], [], [], []]
        boundary_vector_left = self.__create_boundary_vector(Orientation.left)
        boundary_vector_right = self.__create_boundary_vector(Orientation.right)

        # boundary value
        block = np.concatenate((block, boundary_vector_left), axis=1)

        # Create a block in the matrix
        for j in range(i + 1, i + self.__grid_length_x - 1):
            coefficient_vector = self.__create_laplace_operators_matrix_vector(delta_x, delta_y, j)
            block = np.concatenate((block, coefficient_vector), axis=1)

        # boundary value
        block = np.concatenate((block, boundary_vector_right), axis=1)

        return block

    def __create_boundary_block(self, orientation):
        if orientation not in [Orientation.top, Orientation.bottom]:
            raise TypeError("Orientation must be top or bottom")

        if self.__boundary_condition_type == BoundaryConditionsType.dirichlet:
            return self.__create_boundary_block_dirichlet()
        else:
            return self.__create_boundary_block_neumann(orientation)

    def __create_boundary_block_dirichlet(self):
        block = [[], [], [], [], []]
        boundary_vector = self.__create_boundary_vector_dirichlet()
        for i in range(self.__grid_length_x):
            block = np.concatenate((block, boundary_vector), axis=1)
        return block

    def __create_boundary_block_neumann(self, orientation):
        boundary_vector = self.__create_boundary_vector_neumann(orientation)
        block = [[], [], [], [], []]
        for i in range(self.__grid_length_x):
            block = np.concatenate((block, boundary_vector), axis=1)
        return block

    def __create_boundary_vector(self, orientation):
        if orientation not in [Orientation.top, Orientation.bottom, Orientation.left, Orientation.right]:
            raise TypeError("Orientation must be one of top/bottom/left/right")

        if self.__boundary_condition_type == BoundaryConditionsType.dirichlet:
            return self.__create_boundary_vector_dirichlet()
        else:
            return self.__create_boundary_vector_neumann(orientation)

    @staticmethod
    def __create_boundary_vector_dirichlet():
        return [[0], [0], [1], [0], [0]]

    @staticmethod
    def __create_boundary_vector_neumann(orientation):
        if orientation == Orientation.left:
            return [[0], [0], [1], [-1], [0]]
        elif orientation == Orientation.right:
            return [[0], [-1], [1], [0], [0]]
        elif orientation == Orientation.top:
            return [[0], [0], [1], [0], [-1]]
        else:
            return [[-1], [0], [1], [0], [0]]

    def __create_corner_dirichlet_for_neumann(self):
        row = self.__grid_length_x + 1
        self.__operators_matrix[row, :] = 0
        self.__operators_matrix[row, row] = 1

    ###################################### LU decomposition

    def solve(self, solution_matrix):
        if not self.__lu_decomposition_updated:
            self.__operators_matrix_lu_decomposition_update()
        results = self.__lu_operator.solve(solution_matrix.flatten())
        return np.reshape(results, (len(solution_matrix), len(solution_matrix[0])))

    def __operators_matrix_lu_decomposition_update(self):
        self.__lu_operator = sla.splu(self.__operators_matrix)
        self.__lu_decomposition_updated = True

    ###################################### operations on operators matrix

    # TODO: make the function work for single values, arrays and matrix
    # TODO: add possibility to change boundaries, maybe add warnings
    def multiply_operators_matrix(self, multiplier):
        if isinstance(multiplier, numbers.Number):
            self.__multiply_operators_matrix_by_parameter(multiplier)
        elif isinstance(multiplier, list):
            self.__multiply_operators_matrix_by_array(multiplier)

    def __multiply_operators_matrix_by_parameter(self, multiplier):
        multiplier_array = np.empty(0)

        # boundary should not be multiplied
        top_bottom_block_multiplier = np.full([self.__grid_length_x], 1)
        general_block_multiplier = np.concatenate(([1], np.full([self.__grid_length_x - 2], multiplier), [1]))

        # make the array
        multiplier_array = np.append(multiplier_array, top_bottom_block_multiplier)
        num_of_general_blocks = self.__grid_length_y - 2
        for i in range(num_of_general_blocks):
            multiplier_array = np.append(multiplier_array, general_block_multiplier)
        multiplier_array = np.append(multiplier_array, top_bottom_block_multiplier)

        self.__operators_matrix = self.__operators_matrix.multiply(multiplier_array)
        self.__lu_decomposition_updated = False

    def __multiply_operators_matrix_by_array(self, multiplier):
        self.__operators_matrix = self.__operators_matrix.multiply(multiplier)
        self.__lu_decomposition_updated = False

    # TODO: make the function work for single values, arrays and matrix
    # TODO: add possibility to change boundaries, maybe add warnings
    def add_matrix_to_operators_matrix(self, add_matrix):
        # remove rows that changes the boundary
        zero_row = np.full([self.__grid_length_x * self.__grid_length_y], 0)
        num_of_points_in_matrix = self.__grid_length_x * self.__grid_length_y
        for i in range(self.__grid_length_x):
            add_matrix[i] = zero_row
        for i in range(self.__grid_length_x,
                       num_of_points_in_matrix - self.__grid_length_x,
                       self.__grid_length_x):
            add_matrix[i] = zero_row
            add_matrix[i + self.__grid_length_x - 1] = zero_row
        for i in range(self.__grid_length_x):
            add_matrix[num_of_points_in_matrix - i - 1] = zero_row

        self.__operators_matrix += add_matrix
