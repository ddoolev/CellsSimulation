import enum
from general_enums import Field, Orientation
import numpy as np


class BoundaryConditionsType(enum.Enum):
    dirichlet = enum.auto()
    neumann = enum.auto()


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

    @staticmethod
    def add_boundaries(matrix, boundaries, field, orientation, with_side_boundaries=False):
        if orientation == Orientation.all:
            return Boundaries.__add_all_boundaries(matrix, boundaries, field)
        else:
            add_boundaries_function = {
                Orientation.left: Boundaries.__add_left_boundary,
                Orientation.right: Boundaries.__add_right_boundary,
                Orientation.top: Boundaries.__add_top_boundary,
                Orientation.bottom: Boundaries.__add_bottom_boundary,
            }
            function = add_boundaries_function.get(orientation)
            return function(matrix, boundaries, field, with_side_boundaries)

    @staticmethod
    def __add_left_boundary(matrix, boundaries, field, with_top_bottom_boundaries):
        left_boundary = boundaries.get_boundary(Orientation.left, field)
        if with_top_bottom_boundaries:
            left_boundary = np.concatenate(([0], left_boundary, [0]))

        left_boundary = np.array([left_boundary]).T
        return np.append(left_boundary, matrix, axis=1)

    @staticmethod
    def __add_right_boundary(matrix, boundaries, field, with_top_bottom_boundaries):
        right_boundary = boundaries.get_boundary(Orientation.right, field)
        if with_top_bottom_boundaries:
            right_boundary = np.concatenate(([0], right_boundary, [0]))

        right_boundary = np.array([right_boundary]).T
        return np.append(matrix, right_boundary, axis=1)

    @staticmethod
    def __add_bottom_boundary(matrix, boundaries, field, with_left_right_boundaries):
        bottom_boundary = boundaries.get_boundary(Orientation.bottom, field)
        if with_left_right_boundaries:
            bottom_boundary = np.concatenate(([0], bottom_boundary, [0]))

        return np.append([bottom_boundary], matrix, axis=0)

    @staticmethod
    def __add_top_boundary(matrix, boundaries, field, with_left_right_boundaries):
        top_boundary = boundaries.get_boundary(Orientation.top, field)
        if with_left_right_boundaries:
            top_boundary = np.concatenate(([0], top_boundary, [0]))

        return np.append(matrix, [top_boundary], axis=0)

    @staticmethod
    def __add_all_boundaries(matrix, boundaries, field):
        matrix = Boundaries.__add_top_boundary(matrix, boundaries, field, False)
        matrix = Boundaries.__add_bottom_boundary(matrix, boundaries, field, False)
        matrix = Boundaries.__add_left_boundary(matrix, boundaries, field, True)
        matrix = Boundaries.__add_right_boundary(matrix, boundaries, field, True)
        return matrix

