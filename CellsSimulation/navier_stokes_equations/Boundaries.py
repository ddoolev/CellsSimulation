import numpy as np
from typing import Dict
import enum
import warnings


class BoundaryConditionsType(enum.Enum):
    DIRICHLET = 0
    NUEMANN = 1


class Field(enum.Enum):
    U = 0
    V = 1
    P = 2
    ALL = 3


class WarningsStrings:
    NUEMANN_PRESSURE = "Should not use pressure values when in Neumann boundary condition type"


class Boundaries:
    __u: Dict[str:np.array]
    __v: Dict[str:np.array]
    __p: Dict[str:np.array]
    __boundary_conditions_type: BoundaryConditionsType

    def __init__(self, left, right, top, bottom, boundary_conditions_type=BoundaryConditionsType.NUEMANN):
        # Boundaries should not include the corners
        self.__left = left
        self.__right = right
        self.__top = top
        self.__bottom = bottom
        # decide the pressure boundaries type: Dirichlet or Neumann
        self.__boundary_conditions_type = boundary_conditions_type

    @property
    def boundary_conditions_type(self):
        return self.__boundary_conditions_type

    @boundary_conditions_type.setter
    def boundary_conditions_type(self, boundary_conditions_type):
        self.__boundary_conditions_type = boundary_conditions_type

    def get_left(self, field=Field.ALL):
        if field == Field.ALL:
            return self.__left
        elif field == Field.U:
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__left[field]

    def set_left(self, left_boundary, field=Field.ALL):
        if field == Field.ALL:
            self.__left = left_boundary
        else:
            self.__left[field] = left_boundary

    def get_right(self, field=Field.ALL):
        if field == Field.ALL:
            return self.__right
        elif field == Field.U:
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__right[field]

    def set_right(self, right_boundary, field=Field.ALL):
        if field == Field.ALL:
            self.__right = right_boundary
        else:
            self.__right[field] = right_boundary

    def get_top(self, field=Field.ALL):
        if field == Field.ALL:
            return self.__top
        elif field == Field.U:
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__top[field]

    def set_top(self, top_boundary, field=Field.ALL):
        if field == Field.ALL:
            self.__top = top_boundary
        else:
            self.__top[field] = top_boundary

    def get_bottom(self, field=Field.ALL):
        if field == Field.ALL:
            return self.__bottom
        elif field == Field.U:
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__bottom[field]

    def set_bottom(self, bottom_boundary, field=Field.ALL):
        if field == Field.ALL:
            self.__bottom = bottom_boundary
        else:
            self.__bottom[field] = bottom_boundary

