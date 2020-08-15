import numpy as np
import scipy 

class LaplaceOperator:

    def __init__(self, delta_x, delta_y):
        self.__operators_matrix_diagonals = \
            self.__createLaplacOperatorsMatrix(delta_x, delta_y)

    def laplacianOperation(self, data_matrix):
        data_matrix = data_matrix.flatten
        return self.__operators_matrix_diagonals @ data_matrix

    # create Laplac operator's matrix, to calculate the laplacian for every point faster
    
    def __createLaplacOperatorsMatrix(self, delta_x, delta_y):
        grid_length_x = len(delta_x) + 1
        grid_length_y = len(delta_y) + 1
        num_of_points_in_matrix = grid_length_x*grid_length_y
        operators_matrix_diagonals = [[],[],[],[],[]]

        boundary_vector = [[0],[0],[0],[0],[0]]
        boundry_values_block = self.__createBoundryValueBlock(grid_length_x)

        operators_matrix_diagonals = \
            np.concatenate((operators_matrix_diagonals, boundry_values_block), 1)
        
        # Iterate on the matrix blocks
        for i in range(grid_length_x, num_of_points_in_matrix - grid_length_x, grid_length_x):
            block = self.__createLaplacOperatorsMatrixBlock(delta_x, delta_y, i)
            operators_matrix_diagonals = \
                np.concatenate((operators_matrix_diagonals, block), 1)

        operators_matrix_diagonals = \
            np.concatenate((operators_matrix_diagonals, boundry_values_block), 1)

        return operators_matrix_diagonals


    def __createBoundryValueBlock(self, grid_length_x):
        block = [[],[],[],[],[]]
        boundary_vector = [[0],[0],[0],[0],[0]]
        for i in range(grid_length_x):
            block = np.concatenate((block, boundary_vector), 1)
        return block

    
    def __createLaplacOperatorsMatrixVector(self, delta_x, delta_y, i):
        # create the coeficients. 
        # P=plus, M=minus
        grid_length_x = len(delta_x) + 1
        x_index = i//grid_length_x
        y_index = i%grid_length_x

        coe_i_jM1 = 2 / (delta_y[y_index-1] * (delta_y[y_index] + delta_y[y_index-1]))
        coe_iM1_j = 2 / (delta_x[x_index-1] * (delta_x[x_index] + delta_x[x_index-1]))
        coe_i_j = 2 / (delta_x[x_index] * delta_x[x_index-1]) + 2 / (delta_y[y_index] * delta_y[y_index-1])
        coe_iP1_j = 2 / (delta_x[x_index] * (delta_x[x_index] + delta_x[x_index-1]))
        coe_i_jP1 = 2 / (delta_y[y_index] * (delta_y[y_index] + delta_y[y_index-1]))

        coe_vector = [[coe_i_jM1],[coe_iM1_j],[coe_i_j],[coe_iP1_j],[coe_i_jP1]]

        return coe_vector

    
    def __createLaplacOperatorsMatrixBlock(self, delta_x, delta_y, i):
        block = [[],[],[],[],[]]
        boundary_vector = [[0],[0],[0],[0],[0]]
        grid_length_x = len(delta_x) + 1
        # boundary value
        block = np.concatenate((block, boundary_vector), 1)

        # Create a block in the matrix
        for j in range(i+1, i+grid_length_x-1):
            coeficient_vector = self.__createLaplacOperatorsMatrixVector(delta_x, delta_y, j)
            block = np.concatenate((block, coeficient_vector), 1)

        # boundary value
        block = np.concatenate((block, boundary_vector), 1)

        return block