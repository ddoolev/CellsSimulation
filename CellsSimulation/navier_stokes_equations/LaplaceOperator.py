import numpy as np
import scipy.sparse as sparse


class LaplaceOperator:
    __lu_matrix: object

    def __init__(self, delta_x, delta_y):
        self.__operators_matrix = \
            self.__create_laplace_operators_matrix(delta_x, delta_y)
        self.operators_matrix_lu_decomposition()

    def solve(self, solution_matrix):
        results = self.__lu_matrix.solve(solution_matrix)
        return np.reshape(results, (len(solution_matrix), len(solution_matrix[0])))

    def operators_matrix_lu_decomposition(self):
        self.__lu_matrix = sparse.sla.splu(self.__operators_matrix)

    # create Laplace operator's matrix, to calculate the laplacian for every point faster
    def __create_laplace_operators_matrix(self, delta_x, delta_y):
        grid_length_x = len(delta_x) + 1
        grid_length_y = len(delta_y) + 1
        num_of_points_in_matrix = grid_length_x * grid_length_y
        operators_matrix_diagonals = [[], [], [], [], []]

        boundary_values_block = self.__create_boundary_value_block(grid_length_x)

        operators_matrix_diagonals = \
            np.concatenate((operators_matrix_diagonals, boundary_values_block), 1)

        # Iterate on the matrix blocks
        for i in range(grid_length_x, num_of_points_in_matrix - grid_length_x, grid_length_x):
            block = self.__create_laplace_operators_matrix_block(delta_x, delta_y, i)
            operators_matrix_diagonals = \
                np.concatenate((operators_matrix_diagonals, block), 1)

        operators_matrix_diagonals = \
            np.concatenate((operators_matrix_diagonals, boundary_values_block), 1)
        offsets = np.array([grid_length_x, 1, 0, -1, -grid_length_x])

        operators_matrix = sparse.dia_matrix \
            ((operators_matrix_diagonals, offsets),
             shape=(grid_length_x * grid_length_y, grid_length_x * grid_length_y)).transpose()

        return operators_matrix

    @staticmethod
    def __create_boundary_value_block(grid_length_x):
        block = [[], [], [], [], []]
        boundary_vector = [[0], [0], [0], [0], [0]]
        for i in range(grid_length_x):
            block = np.concatenate((block, boundary_vector), 1)
        return block

    @staticmethod
    def __create_laplace_operators_matrix_vector(delta_x, delta_y, i):
        # create the coefficients.
        # P=plus, M=minus
        grid_length_x = len(delta_x) + 1
        x_index = i // grid_length_x
        y_index = i % grid_length_x

        coe_i_jM1 = 2 / (delta_y[y_index - 1] * (delta_y[y_index] + delta_y[y_index - 1]))
        coe_iM1_j = 2 / (delta_x[x_index - 1] * (delta_x[x_index] + delta_x[x_index - 1]))
        coe_i_j = -(2 / (delta_x[x_index] * delta_x[x_index - 1]) + 2 / (delta_y[y_index] * delta_y[y_index - 1]))
        coe_iP1_j = 2 / (delta_x[x_index] * (delta_x[x_index] + delta_x[x_index - 1]))
        coe_i_jP1 = 2 / (delta_y[y_index] * (delta_y[y_index] + delta_y[y_index - 1]))

        coe_vector = [[coe_i_jM1], [coe_iM1_j], [coe_i_j], [coe_iP1_j], [coe_i_jP1]]

        return coe_vector

    def __create_laplace_operators_matrix_block(self, delta_x, delta_y, i):
        block = [[], [], [], [], []]
        boundary_vector = [[0], [0], [0], [0], [0]]
        grid_length_x = len(delta_x) + 1
        # boundary value
        block = np.concatenate((block, boundary_vector), 1)

        # Create a block in the matrix
        for j in range(i + 1, i + grid_length_x - 1):
            coefficient_vector = self.__create_laplace_operators_matrix_vector(delta_x, delta_y, j)
            block = np.concatenate((block, coefficient_vector), 1)

        # boundary value
        block = np.concatenate((block, boundary_vector), 1)

        return block

    def multiply_operators_matrix(self, argument):
        self.__operators_matrix = self.__operators_matrix.multiply(argument)

    def add_to_operators_matrix(self, add_matrix):
        self.__operators_matrix += add_matrix
