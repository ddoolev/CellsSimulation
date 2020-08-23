import numpy as np
import scipy 

class NavierStokesEquations:

    def __init__(self, data_matrix, delta_x, delta_y):
        self.__operators_matrix_diagonals = \
            __createLaplacOperatorsMatrix(self, data_matrix, delta_x, delta_y)