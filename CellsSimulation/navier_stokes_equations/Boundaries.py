import numpy as np
from typing import Dict
import enum
import warnings


class BOUNDARY_CONDITIONS_TYPE(enum.Enum):
    DIRICHLET = 0
    NUEMANN = 1


class FIELD(enum.Enum):
    U = 0
    V = 1
    P = 2


class WarningsStrings:
    NUEMANN_PRESSURE = "Should not use pressure values when in Neumann boundary condition type"


class Boundaries:
    __u: Dict[str:np.array]
    __v: Dict[str:np.array]
    __p: Dict[str:np.array]
    __boundary_conditions_type: BOUNDARY_CONDITIONS_TYPE

    def __init__(self, left, right, top, bottom, boundary_conditions_type=BOUNDARY_CONDITIONS_TYPE.NUEMANN):
        # Boundaries should not include the corners
        self.__u = u
        self.__v = v
        # decide the pressure boundaries type: Dirichlet or Neumann
        self.__boundary_conditions_type = boundary_conditions_type

        if self.boundary_conditions_type == BOUNDARY_CONDITIONS_TYPE.DIRICHLET:
            self.__p = p
        else:
            self.__p = {SIDE.LEFT: [], SIDE.RIGHT: [], SIDE.BOTTOM: [], SIDE.TOP: []}

    @property
    def boundary_conditions_type(self):
        return self.__boundary_conditions_type

    @boundary_conditions_type.setter
    def boundary_conditions_type(self, boundary_conditions_type):
        self.__boundary_conditions_type = boundary_conditions_type

    def get_left(self, field):
        return self.__u[side]

    def set_u(self, u_boundary, side=SIDE.ALL):
        if side == SIDE.ALL:
            self.__u = u_boundary
        else:
            self.__u[side] = u_boundary

    def get_v(self, side=SIDE.ALL):
        if side == SIDE.ALL:
            return self.__v
        return self.__v[side]

    def set_v(self, v_boundary, side=SIDE.ALL):
        if side == SIDE.ALL:
            self.__v = v_boundary
        else:
            self.__v[side] = v_boundary

    def get_p(self, side=SIDE.ALL):
        if self.boundary_conditions_type == BOUNDARY_CONDITIONS_TYPE.DIRICHLET:
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        if side == SIDE.ALL:
            return self.__p
        return self.__p[side]

    def set_p(self, p_boundary, side=SIDE.ALL):
        if side == SIDE.ALL:
            self.__p = p_boundary
        else:
            self.__p[side] = p_boundary
