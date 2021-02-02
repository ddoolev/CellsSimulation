
import numpy as np
import General_enums as E


class WarningsStrings:
    neumann_pressure = "Should not use pressure values when in Neumann boundary condition type"


class Boundaries:

    def __init__(self, boundaries):
        self.__boundaries = boundaries

    def get_boundary(self, orientation, field=E.Field.all):
        if field == E.Field.all:
            return self.__boundaries[orientation]
        return self.__boundaries[orientation][field]

    def set_boundary(self, boundary, orientation, field=E.Field.all):
        if field == E.Field.all:
            self.__boundaries[orientation] = boundary
        else:
            self.__boundaries[orientation][field] = boundary

    @staticmethod
    def remove_side(array, orientation):
        if array.ndim == 1:
            return Boundaries.__remove_array_side(array, orientation)
        elif array.ndim == 2:
            return Boundaries.__remove_matrix_side(array, orientation)
        else:
            raise TypeError("Can not handle array with more than 2 dimensions")

    @staticmethod
    def __remove_matrix_side(matrix, orientation):
        remove_boundary = {
            E.Orientation.top: Boundaries.__remove_matrix_side_top(matrix),
            E.Orientation.bottom: Boundaries.__remove_matrix_side_bottom(matrix),
            E.Orientation.left: Boundaries.__remove_matrix_side_left(matrix),
            E.Orientation.right: Boundaries.__remove_matrix_side_right(matrix),
            E.Orientation.all: Boundaries.__remove_matrix_side_all(matrix)
        }

        return remove_boundary.get(orientation)

    @staticmethod
    def __remove_array_side(array, orientation):
        remove_boundary = {
            E.Orientation.left: Boundaries.__remove_array_side_left(array),
            E.Orientation.right: Boundaries.__remove_array_side_right(array),
            E.Orientation.all: Boundaries.__remove_array_side_all(array)
        }

        return remove_boundary.get(orientation)

    @staticmethod
    def __remove_matrix_side_top(matrix):
        return matrix[:-1, :]

    @staticmethod
    def __remove_matrix_side_bottom(matrix):
        return matrix[1:, :]

    @staticmethod
    def __remove_matrix_side_left(matrix):
        return matrix[:, 1:]

    @staticmethod
    def __remove_matrix_side_right(matrix):
        return matrix[:, :-1]

    @staticmethod
    def __remove_matrix_side_all(matrix):
        return matrix[1:-1, 1:-1]

    @staticmethod
    def __remove_array_side_all(array):
        return array[1:-1]

    @staticmethod
    def __remove_array_side_left(array):
        return array[1:]

    @staticmethod
    def __remove_array_side_right(array):
        return array[:-1]

    @staticmethod
    def add_boundaries(matrix, boundaries, field, orientation, with_edge_boundaries=False):
        if orientation == E.Orientation.all:
            return Boundaries.__add_all_boundaries(matrix, boundaries, field)
        else:
            add_boundaries_function = {
                E.Orientation.left: Boundaries.__add_left_boundary,
                E.Orientation.right: Boundaries.__add_right_boundary,
                E.Orientation.top: Boundaries.__add_top_boundary,
                E.Orientation.bottom: Boundaries.__add_bottom_boundary,
            }
            function = add_boundaries_function.get(orientation)
            return function(matrix, boundaries, field, with_edge_boundaries)

    @staticmethod
    def __add_left_boundary(matrix, boundaries, field, with_top_bottom_boundaries):
        left_boundary = boundaries.get_boundary(E.Orientation.left, field)
        if with_top_bottom_boundaries:
            left_boundary = np.concatenate(([0], left_boundary, [0]))

        left_boundary = np.array([left_boundary]).T
        return np.append(left_boundary, matrix, axis=1)

    @staticmethod
    def __add_right_boundary(matrix, boundaries, field, with_top_bottom_boundaries):
        right_boundary = boundaries.get_boundary(E.Orientation.right, field)
        if with_top_bottom_boundaries:
            right_boundary = np.concatenate(([0], right_boundary, [0]))

        right_boundary = np.array([right_boundary]).T
        return np.append(matrix, right_boundary, axis=1)

    @staticmethod
    def __add_bottom_boundary(matrix, boundaries, field, with_left_right_boundaries):
        bottom_boundary = boundaries.get_boundary(E.Orientation.bottom, field)
        if with_left_right_boundaries:
            bottom_boundary = np.concatenate(([0], bottom_boundary, [0]))

        return np.append([bottom_boundary], matrix, axis=0)

    @staticmethod
    def __add_top_boundary(matrix, boundaries, field, with_left_right_boundaries):
        top_boundary = boundaries.get_boundary(E.Orientation.top, field)
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
