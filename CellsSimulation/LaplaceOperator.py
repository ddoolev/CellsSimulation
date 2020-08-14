import numpy as np
import scipy 

class LaplaceOperator:

    def __init__(self, data_matrix, delta_x, delta_y):
        self.__operators_matrix_diagonals = \
            __createLaplacOperatorsMatrix(self, data_matrix, delta_x, delta_y)

    def laplacianOperation(self, data_matrix):
        return self.__operators_matrix_diagonals @ data_matrix

    # create Laplac operator's matrix, to calculate the laplacian for every point faster
    @classmethod
    def __createLaplacOperatorsMatrix(self, data_matrix, delta_x, delta_y):
        num_of_points = len(data_matrix)*len(data_matrix[0])
        operators_matrix_diagonals = [[],[],[],[],[]]

        boundary_vector = [[0],[0],[0],[0],[0]]
        boundry_values_block = self.__createBoundryValueBlock()

        operators_matrix_diagonals = \
            np.concatenate((operators_matrix_diagonals, boundry_values_block), 1)
        
        # Iterate on the matrix blocks
        for i in range(len(data_matrix), num_of_points - len(data_matrix), len(data_matrix)):
            block = self.__createLaplacOperatorsMatrixBlock(delta_x, delta_y)
            operators_matrix_diagonals = \
                np.concatenate((operators_matrix_diagonals, block), 1)

        operators_matrix_diagonals = \
            np.concatenate((operators_matrix_diagonals, boundry_values_block), 1)

    @classmethod
    def __createBoundryValueBlock(self):
        block = [[],[],[],[],[]]
        boundary_vector = [[0],[0],[0],[0],[0]]
        for i in range(len(data_matrix)):
            block = np.concatenate((block, boundary_vector), 1)
        return block

    @classmethod
    def __createLaplacOperatorsMatrixVector(self, delta_x, delta_y):
        # create the coeficients. 
        # P=plus, M=minus
        coe_i_jM1 = 2 / (delta_y[i-1] * (delta_y[i] + delta_y[i-1]))
        coe_iM1_j = 2 / (delta_x[i-1] * (delta_x[i] + delta_x[i-1]))
        coe_i_j = 2 / (delta_x[i] * delta_x[i-1]) + 2 / (delta_y[i] * delta_y[i-1])
        coe_iP1_j = 2 / (delta_x[i] * (delta_x[i] + delta_x[i-1]))
        coe_i_jP1 = 2 / (delta_y[i] * (delta_y[i] + delta_y[i-1]))

        coe_vector = [[coe_i_jM1],[coe_iM1_j],[coe_i_j],[coe_iP1_j],[coe_i_jP1]]

        return coe_vector

    @classmethod
    def __createLaplacOperatorsMatrixBlock(self, delta_x, deltay):
        block = [[],[],[],[],[]]
        # boundary value
        block = np.concatenate((block, boundary_vector), 1)

        # Create a block in the matrix
        for j in range(i+1, i+len(data_matrix)-1):
            coe_vector = self.__createLaplacOperatorsMatrixVector(delta_x,delta_y)
            block = np.concatenate((block, coe_vector), 1)

        # boundary value
        block = np.concatenate((block, boundary_vector), 1)

        return block