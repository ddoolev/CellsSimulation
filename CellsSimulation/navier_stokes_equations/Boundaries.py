import enum
from general_enums import Field, Orientation


class BoundaryConditionsType(enum.Enum):
    dirichlet = 0
    neumann = 1


class WarningsStrings:
    neumann_pressure = "Should not use pressure values when in Neumann boundary condition type"


class Boundaries:

    def __init__(self, boundaries):
        self.__boundaries = boundaries

    def get_boundary(self, orientation, field=Field.all):
        if field == Field.all:
            return self.__boundaries[orientation]
        return self.__boundaries[orientation][field]

    def set_boundary(self, boundary, orientation, field=Field.all):
        if field == Field.all:
            self.__boundaries[orientation] = boundary
        else:
            self.__boundaries[orientation][field] = boundary

    @staticmethod
    def remove_boundaries(array, orientation):
        if array.ndim == 1:
            return Boundaries.__remove_array_boundaries(array, orientation)
        elif array.ndim == 2:
            return Boundaries.__remove_matrix_boundaries(array, orientation)
        else:
            raise TypeError("Can not handle array with more than 2 dimensions")

    @staticmethod
    def __remove_matrix_boundaries(matrix, orientation):
        remove_boundary = {
            Orientation.top: Boundaries.__remove_matrix_boundaries_top(matrix),
            Orientation.bottom: Boundaries.__remove_matrix_boundaries_bottom(matrix),
            Orientation.left: Boundaries.__remove_matrix_boundaries_left(matrix),
            Orientation.right: Boundaries.__remove_matrix_boundaries_right(matrix),
            Orientation.all: Boundaries.__remove_matrix_boundaries_all(matrix)
        }

        return remove_boundary.get(orientation)

    @staticmethod
    def __remove_array_boundaries(array, orientation):
        remove_boundary = {
            Orientation.left: Boundaries.__remove_array_boundaries_left(array),
            Orientation.right: Boundaries.__remove_array_boundaries_right(array),
            Orientation.all: Boundaries.__remove_array_boundaries_all(array)
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

