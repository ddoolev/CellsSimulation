import numpy as np
from typing import Dict
import enum
import warnings


class BoundaryConditionsType(enum.Enum):
    DIRICHLET = 0
    NEUMANN = 1


class WarningsStrings:
    NUEMANN_PRESSURE = "Should not use pressure values when in Neumann boundary condition type"


class Boundaries:
    __u: Dict[str:np.array]
    __v: Dict[str:np.array]
    __p: Dict[str:np.array]
    __boundary_conditions_type: BoundaryConditionsType

    def __init__(self, boundaries, boundary_conditions_type):
        # The boundaries should look like this:
        # boundaries dictionary: {"left":[], "right":[], "top":[], "bottom":[]}
        # Boundaries should not include the corners
        self.__u = boundaries["u"]
        self.__v = boundaries["v"]
        # decide the pressure boundaries type: Dirichlet or Neumann
        self.__boundary_conditions_type = boundary_conditions_type

        if self.is_boundary_condition_type_dirichlet():
            self.__p = boundaries["p"]
        else:
            self.__p = {"left": [], "right": [], "top": [], "bottom": []}

    def is_boundary_condition_type_dirichlet(self):
        return self.__boundary_conditions_type == BoundaryConditionsType.DIRICHLET

    @property
    def boundary_conditions_type(self):
        return self.__boundary_conditions_type

    @boundary_conditions_type.setter
    def boundary_conditions_type(self, boundary_conditions_type):
        self.__boundary_conditions_type = boundary_conditions_type

    @property
    def u(self):
        return self.__u

    @u.setter
    def u(self, u_boundary):
        self.__u = u_boundary

    @property
    def u_left(self):
        return self.__u["left"]

    @u_left.setter
    def u_left(self, u_left_boundary):
        self.__u["left"] = u_left_boundary

    @property
    def u_top(self):
        return self.__u["top"]

    @u_top.setter
    def u_top(self, u_top_boundary):
        self.__u["top"] = u_top_boundary

    @property
    def u_right(self):
        return self.__u["right"]

    @u_right.setter
    def u_right(self, u_right_boundary):
        self.__u["right"] = u_right_boundary

    @property
    def u_bottom(self):
        return self.__u["bottom"]

    @u_bottom.setter
    def u_bottom(self, u_bottom_boundary):
        self.__u["bottom"] = u_bottom_boundary

    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, v_boundary):
        self.__v = v_boundary

    @property
    def v_left(self):
        return self.__v["left"]

    @v_left.setter
    def v_left(self, v_left_boundary):
        self.__v["left"] = v_left_boundary

    @property
    def v_top(self):
        return self.__v["top"]

    @v_top.setter
    def v_top(self, v_top_boundary):
        self.__v["top"] = v_top_boundary

    @property
    def v_right(self):
        return self.__v["right"]

    @v_right.setter
    def v_right(self, v_right_boundary):
        self.__v["right"] = v_right_boundary

    @property
    def v_bottom(self):
        return self.__v["bottom"]

    @v_bottom.setter
    def v_bottom(self, v_bottom_boundary):
        self.__v["bottom"] = v_bottom_boundary

    @property
    def p(self):
        if not self.is_boundary_condition_type_dirichlet():
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__p

    @p.setter
    def p(self, p_boundary):
        self.__p = p_boundary

    @property
    def p_left(self):
        if not self.is_boundary_condition_type_dirichlet():
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__p["left"]

    @p_left.setter
    def p_left(self, p_left_boundary):
        self.__p["left"] = p_left_boundary

    @property
    def p_top(self):
        if not self.is_boundary_condition_type_dirichlet():
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__p["top"]

    @p_top.setter
    def p_top(self, p_top_boundary):
        self.__p["top"] = p_top_boundary

    @property
    def p_right(self):
        if not self.is_boundary_condition_type_dirichlet():
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__p["right"]

    @p_right.setter
    def p_right(self, p_right_boundary):
        self.__p["right"] = p_right_boundary

    @property
    def p_bottom(self):
        if not self.is_boundary_condition_type_dirichlet():
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__p["bottom"]

    @p_bottom.setter
    def p_bottom(self, p_bottom_boundary):
        self.__p["bottom"] = p_bottom_boundary
