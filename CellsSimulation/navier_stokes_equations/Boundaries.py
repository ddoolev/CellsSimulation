import enum
import warnings
from enums import Fields


class BoundaryConditionsType(enum.Enum):
    DIRICHLET = 0
    NUEMANN = 1


class WarningsStrings:
    NUEMANN_PRESSURE = "Should not use pressure values when in Neumann boundary condition type"


class Boundaries:
    # __u: Dict[str:np.array]
    # __v: Dict[str:np.array]
    # __p: Dict[str:np.array]
    # __boundary_conditions_type: BoundaryConditionsType

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

    def get_left(self, fields=Fields.ALL):
        if fields == fields.ALL:
            return self.__left
        elif fields == fields.U:
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__left[fields]

    def set_left(self, left_boundary, fields=Fields.ALL):
        if fields == Fields.ALL:
            self.__left = left_boundary
        else:
            self.__left[fields] = left_boundary

    def get_right(self, fields=Fields.ALL):
        if fields == Fields.ALL:
            return self.__right
        elif fields == Fields.U:
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__right[fields]

    def set_right(self, right_boundary, fields=Fields.ALL):
        if fields == Fields.ALL:
            self.__right = right_boundary
        else:
            self.__right[fields] = right_boundary

    def get_top(self, fields=Fields.ALL):
        if fields == Fields.ALL:
            return self.__top
        elif fields == Fields.U:
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__top[fields]

    def set_top(self, top_boundary, fields=Fields.ALL):
        if fields == Fields.ALL:
            self.__top = top_boundary
        else:
            self.__top[fields] = top_boundary

    def get_bottom(self, fields=Fields.ALL):
        if fields == Fields.ALL:
            return self.__bottom
        elif fields == Fields.U:
            warnings.warn(WarningsStrings.NUEMANN_PRESSURE)
        return self.__bottom[fields]

    def set_bottom(self, bottom_boundary, fields=Fields.ALL):
        if fields == Fields.ALL:
            self.__bottom = bottom_boundary
        else:
            self.__bottom[fields] = bottom_boundary
