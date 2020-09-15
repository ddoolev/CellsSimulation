import enum
import warnings
from NavierStokesEquations import Fields, Delta
import numpy as np


class BoundaryConditionsType(enum.Enum):
    DIRICHLET = 0
    NEUMANN = 1


class WarningsStrings:
    NEUMANN_PRESSURE = "Should not use pressure values when in Neumann boundary condition type"


class Boundaries:
    # __u: Dict[str:np.array]
    # __v: Dict[str:np.array]
    # __p: Dict[str:np.array]
    # __boundary_conditions_type: BoundaryConditionsType

    def __init__(self, boundaries, boundary_conditions_type=BoundaryConditionsType.neumann):
        self.__boundaries = boundaries
        # decide the pressure boundaries type: Dirichlet or Neumann
        self.__boundary_conditions_type = boundary_conditions_type

    @property
    def boundary_conditions_type(self):
        return self.__boundary_conditions_type

    @boundary_conditions_type.setter
    def boundary_conditions_type(self, boundary_conditions_type):
        self.__boundary_conditions_type = boundary_conditions_type

    def get_boundary(self, orientation, field=Fields.all):
        if field == Fields.all:
            return self.__boundaries[orientation]
        elif field == Fields.p and self.__boundary_conditions_type == BoundaryConditionsType.neumann:
            warnings.warn(WarningsStrings.neumann_pressure)
        return self.__boundaries[orientation][field]

    def set_boundary(self, boundary, orientation, field=Fields.all):
        if field == Fields.all:
            self.__boundaries[orientation] = boundary
        else:
            self.__boundaries[orientation][field] = boundary

    def add_boundaries_left(self, matrix, field):
        if field == Fields.p and self.__boundaries.boundary_conditions_type == BoundaryConditionsType.neumann:
            left_boundary = np.concatenate(([0], matrix[0], [0]))
        else:
            left_boundary = np.concatenate(([0], self.__boundaries.get_boundary(Orientation.left, field), [0]), axis=0)
        left_boundary = np.array([left_boundary]).T
        return np.append([left_boundary], matrix, axis=1)

    def add_boundaries_right(self, matrix, field):
        if field == Fields.p and self.__boundaries.boundary_conditions_type == BoundaryConditionsType.neumann:
            right_boundary = np.concatenate(([0], matrix[-1], [0]))
        else:
            right_boundary = np.concatenate(([0], self.__boundaries.get_boundary(Orientation.right, field), [0]), axis=0)
        right_boundary = np.array([right_boundary]).T
        return np.append(matrix, [right_boundary], axis=1)

    def add_boundaries_top(self, matrix, field):
        if field == Fields.p and self.__boundaries.boundary_conditions_type == BoundaryConditionsType.neumann:
            top_boundary = np.concatenate(([0], matrix.T[0], [0]))
        else:
            top_boundary = np.concatenate(([0], self.__boundaries.get_boundary(Orientation.top, field), [0]), axis=0)
        return np.append([top_boundary], matrix, axis=0)

    def add_boundaries_bottom(self, matrix, field):
        if field == Fields.p and self.__boundaries.boundary_conditions_type == BoundaryConditionsType.neumann:
            bottom_boundary = np.concatenate(([0], matrix.T[-1], [0]))
        else:
            bottom_boundary = np.concatenate(([0], self.__boundaries.get_boundary(Orientation.bottom, field), [0]), axis=0)
        return np.append(matrix, bottom_boundary, axis=0)

    def add_boundaries_all(self, matrix, field):
        if field == Fields.p and self.__boundaries.boundary_conditions_type == BoundaryConditionsType.neumann:
            left_boundary = np.array([matrix[0]]).T
            right_boundary = np.array([matrix[-1]]).T
            top_boundary = np.concatenate(([0], matrix[0].T, [0]), axis=0)
            bottom_boundary = np.concatenate(([0], matrix[-1].T, [0]), axis=0)
        else:
            left_boundary = np.array([self.__boundaries.get_boundary(Orientation.left, field)]).T
            right_boundary = np.array([self.__boundaries.get_boundary(Orientation.right, field)]).T
            top_boundary = np.concatenate(([0], self.__boundaries.get_boundary(Orientation.top, field), [0]), axis=0)
            bottom_boundary = np.concatenate(([0], self.__boundaries.get_boundary(Orientation.bottom, field), [0]), axis=0)
        top_boundary = np.array([top_boundary])
        bottom_boundary = np.array([bottom_boundary])

        matrix = np.concatenate((left_boundary, matrix, right_boundary), axis=1)
        matrix = np.concatenate((top_boundary, matrix, bottom_boundary), axis=0)
        return matrix

