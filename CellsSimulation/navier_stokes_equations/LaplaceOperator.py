import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg as sla
import GeneralEnums as E
import numbers


class LaplaceOperator:

    def __init__(self, delta, field, boundary_condition_type):
        self.__delta = delta
        self.__boundary_condition_type = boundary_condition_type
        self.__operators_matrix_updated_flag = True
        self.__field = field
        if field == E.Field.all:
            raise TypeError("Field can't be of type 'all'")
        self.__operators_matrix = \
            self.__create_laplace_operators_matrix()

    def __create_laplace_operators_matrix(self):
        self.__get_delta_grid_length()
        matrix_side_size = self.__grid_length_x * self.__grid_length_y
        operators_matrix_diagonals = [[], [], [], [], []]

        boundary_block_top = self.__create_boundary_diagonal_block(E.Orientation.top)
        boundary_block_bottom = self.__create_boundary_diagonal_block(E.Orientation.bottom)

        operators_matrix_diagonals = \
            np.concatenate((boundary_block_top, operators_matrix_diagonals), axis=1)

        # Iterate on the matrix blocks
        for i in range(self.__grid_length_x,
                       matrix_side_size - self.__grid_length_x,
                       self.__grid_length_x):
            block = self.__create_laplace_operators_matrix_block(i)
            operators_matrix_diagonals = \
                np.concatenate((operators_matrix_diagonals, block), axis=1)

        operators_matrix_diagonals = \
            np.concatenate((operators_matrix_diagonals, boundary_block_bottom), axis=1)

        offsets = np.array([self.__grid_length_x, 1, 0, -1, -self.__grid_length_x])

        operators_matrix = sparse.spdiags(
            operators_matrix_diagonals, offsets,
            matrix_side_size, matrix_side_size,
            format='csc').T
        np.set_printoptions(threshold=1000)
        return operators_matrix

    @staticmethod
    def __add_block_to_operators_matrix_diagonals(operators_matrix_diagonals, block):
        return np.concatenate((operators_matrix_diagonals, block), axis=1)

    def __get_delta_grid_length(self):
        if self.__field == E.Field.u:
            self.__grid_length_x = len(self.__delta.delta_x) + 1
            self.__grid_length_y = len(self.__delta.delta_half_y) + 1
        elif self.__field == E.Field.v:
            self.__grid_length_x = len(self.__delta.delta_half_x) + 1
            self.__grid_length_y = len(self.__delta.delta_y) + 1
        else:  # self.__field == E.Field.p:
            self.__grid_length_x = len(self.__delta.delta_half_x) + 1
            self.__grid_length_y = len(self.__delta.delta_half_y) + 1

    def __create_laplace_operators_matrix_vector(self, i):
        # create the coefficients.
        # P=plus, M=minus, h=half
        x_index = i % self.__grid_length_x - 1
        y_index = i // self.__grid_length_x - 1
        x_index_half_grid = x_index + 1
        y_index_half_grid = y_index + 1

        if self.__field == E.Field.u:
            coe_i_jM1 = self.__delta.delta_half_x[x_index_half_grid] / self.__delta.delta_half_y[y_index_half_grid - 1]
            coe_iM1_j = self.__delta.delta_y[y_index] / self.__delta.delta_x[x_index - 1]
            coe_iP1_j = self.__delta.delta_y[y_index] / self.__delta.delta_x[x_index]
            coe_i_jP1 = self.__delta.delta_half_x[x_index_half_grid] / self.__delta.delta_half_y[y_index_half_grid]
        elif self.__field == E.Field.v:
            coe_i_jM1 = self.__delta.delta_x[x_index] / self.__delta.delta_y[y_index - 1]
            coe_iM1_j = self.__delta.delta_half_y[y_index_half_grid] / self.__delta.delta_half_x[x_index_half_grid - 1]
            coe_iP1_j = self.__delta.delta_half_y[y_index_half_grid] / self.__delta.delta_half_x[x_index_half_grid]
            coe_i_jP1 = self.__delta.delta_x[x_index] / self.__delta.delta_y[y_index]
        else:  # self.__field == E.Field.p
            coe_i_jM1 = 1 / self.__delta.delta_half_y[y_index_half_grid - 1] ** 2
            coe_iM1_j = 1 / self.__delta.delta_half_x[x_index_half_grid - 1] ** 2
            coe_iP1_j = 1 / self.__delta.delta_half_x[x_index_half_grid] ** 2
            coe_i_jP1 = 1 / self.__delta.delta_half_y[y_index_half_grid] ** 2

        coe_i_j = -(coe_i_jM1 + coe_iM1_j + coe_iP1_j + coe_i_jP1)

        coe_vector = [[coe_i_jM1], [coe_iM1_j], [coe_i_j], [coe_iP1_j], [coe_i_jP1]]

        return coe_vector

    def __create_laplace_operators_matrix_block(self, i):
        block = [[], [], [], [], []]
        boundary_vector_left = self.__create_boundary_vector_diagonal(E.Orientation.left)
        boundary_vector_right = self.__create_boundary_vector_diagonal(E.Orientation.right)

        # boundary value
        block = np.concatenate((boundary_vector_left, block ), axis=1)

        # Create a block in the matrix
        for j in range(i + 1, i + self.__grid_length_x - 1):
            coefficient_vector = \
                self.__create_laplace_operators_matrix_vector(j)
            block = np.concatenate((block, coefficient_vector), axis=1)

        # boundary value
        block = np.concatenate((block, boundary_vector_right), axis=1)

        return block

    ###################################### Create Boundaries block and vector

    def __create_boundary_diagonal_block(self, orientation):
        if orientation not in [E.Orientation.top, E.Orientation.bottom]:
            raise TypeError("Orientation must be top or bottom")

        if self.__boundary_condition_type == E.BoundaryConditionsType.dirichlet:
            return self.__create_boundary_block_diagonal_dirichlet()
        else:
            return self.__create_boundary_block_diagonal_neumann(orientation)

    def __create_boundary_block_diagonal_dirichlet(self):
        block = [[], [], [], [], []]
        boundary_vector = self.__create_boundary_vector_diagonal_dirichlet()
        for i in range(self.__grid_length_x):
            block = np.concatenate((block, boundary_vector), axis=1)
        return block

    def __create_boundary_block_diagonal_neumann(self, orientation):
        boundary_vector = self.__create_boundary_vector_diagonal_neumann(orientation)
        block = [[], [], [], [], []]
        for i in range(self.__grid_length_x):
            block = np.concatenate((block, boundary_vector), axis=1)
        return block

    def __create_boundary_vector(self, orientation, row):
        diagonal_vector = self.__create_boundary_vector_diagonal(orientation)
        return self.__diagonal_vector_to_normal_vector(diagonal_vector, row)

    def __diagonal_vector_to_normal_vector(self, diagonal_vector, row):
        num_of_points_in_matrix = self.__grid_length_x * self.__grid_length_y
        vector = np.full([num_of_points_in_matrix], 0)
        vector[row] = diagonal_vector[2][0]
        if row > 0:
            vector[row-1] = diagonal_vector[1][0]
            if row >= self.__grid_length_x:
                vector[row - self.__grid_length_x] = diagonal_vector[0][0]
        if row < num_of_points_in_matrix - 1:
            vector[row+1] = diagonal_vector[3][0]
            if row < num_of_points_in_matrix - self.__grid_length_x:
                vector[row + self.__grid_length_x] = diagonal_vector[4][0]
        return vector

    def __create_boundary_vector_diagonal(self, orientation):
        if orientation not in [E.Orientation.top, E.Orientation.bottom, E.Orientation.left, E.Orientation.right]:
            raise TypeError("Orientation must be one of top/bottom/left/right")

        if self.__boundary_condition_type == E.BoundaryConditionsType.dirichlet:
            return LaplaceOperator.__create_boundary_vector_diagonal_dirichlet()
        else:
            return LaplaceOperator.__create_boundary_vector_diagonal_neumann(orientation)

    @staticmethod
    def __create_boundary_vector_diagonal_dirichlet():
        return [[0], [0], [1], [0], [0]]

    @staticmethod
    def __create_boundary_vector_diagonal_neumann(orientation):
        if orientation == E.Orientation.left:
            return [[0], [0], [1], [-1], [0]]
        elif orientation == E.Orientation.right:
            return [[0], [-1], [1], [0], [0]]
        elif orientation == E.Orientation.top:
            return [[0], [0], [1], [0], [-1]]
        else:
            return [[-1], [0], [1], [0], [0]]

    def __create_corner_dirichlet_for_neumann(self):
        row = self.__grid_length_x + 1
        self.__operators_matrix[row, :] = 0
        self.__operators_matrix[row, row] = 1

    ###################################### LU decomposition

    def solve(self, solution_matrix):
        self.__check_operators_matrix_update()
        results = self.__lu_operator.solve(solution_matrix.flatten())
        return np.reshape(results, (len(solution_matrix), len(solution_matrix[0])))

    def __check_operators_matrix_update(self):
        if self.__operators_matrix_updated_flag:
            self.__redo_operators_matrix_boundaries()
            if self.__boundary_condition_type == E.BoundaryConditionsType.neumann:
                self.__create_corner_dirichlet_for_neumann()
            self.__lu_operator = sla.splu(self.__operators_matrix)
            self.__operators_matrix_updated_flag = False

    def __redo_operators_matrix_boundaries(self):
        num_of_points_in_matrix = self.__grid_length_x * self.__grid_length_y
        for i in range(self.__grid_length_x):
            self.__operators_matrix[i] = self.__create_boundary_vector(E.Orientation.top, i)
        for i in range(self.__grid_length_x,
                       num_of_points_in_matrix - self.__grid_length_x,
                       self.__grid_length_x):
            self.__operators_matrix[i] = self.__create_boundary_vector(E.Orientation.left, i)
            self.__operators_matrix[i + self.__grid_length_x - 1] = \
                self.__create_boundary_vector(E.Orientation.right, i + self.__grid_length_x - 1)

        for i in range(self.__grid_length_x):
            self.__operators_matrix[num_of_points_in_matrix - i - 1] = \
                self.__create_boundary_vector(E.Orientation.bottom, num_of_points_in_matrix - i - 1)

    ###################################### operations on operators matrix

    def multiply_operators_matrix(self, multiplier, transpose=False):
        if isinstance(multiplier, numbers.Number):
            self.__multiply_operators_matrix_by_parameter(multiplier)
        elif isinstance(multiplier, np.ndarray):
            if multiplier.ndim == 1:
                self.__multiply_operators_matrix_by_array(multiplier, transpose)
            elif multiplier.ndim == 2:
                self.__multiply_operators_matrix_by_matrix(multiplier)

        self.__operators_matrix_updated_flag = True

    def __multiply_operators_matrix_by_parameter(self, multiplier):
        self.__operators_matrix = self.__operators_matrix.multiply(multiplier)

    def __multiply_operators_matrix_by_array(self, multiplier, transpose=False):
        if transpose:
            self.__operators_matrix = (self.__operators_matrix.T.multiply(multiplier)).T
        else:
            self.__operators_matrix = self.__operators_matrix.multiply(multiplier)

    def __multiply_operators_matrix_by_matrix(self, multiplier):
        self.__operators_matrix = self.__operators_matrix.multiply(multiplier)

    # TODO: make the function work for single values, arrays and matrix
    def add_matrix_to_operators_matrix(self, add_matrix):
        self.__operators_matrix += add_matrix
        self.__operators_matrix_updated_flag = True

    def print(self):
        np.set_printoptions(100)
        self.__print_matrix(np.array(self.__operators_matrix))

    @staticmethod
    def __print_matrix(matrix):
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix]))
        print("\n\n")
