import enum
from general_enums import Field


class BoundaryConditionsType(enum.Enum):
    dirichlet = 0
    neumann = 1


class WarningsStrings:
    neumann_pressure = "Should not use pressure values when in Neumann boundary condition type"


class Boundaries:
    # __u: Dict[str:np.array]
    # __v: Dict[str:np.array]
    # __p: Dict[str:np.array]
    # __boundary_conditions_type: BoundaryConditionsType

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

    # def add_top_bottom(self, matrix, field, with_left_right_boundaries=False):
    #     if with_left_right_boundaries:
    #         top = [np.concatenate(([0], self.get_boundary(Orientation.top, field), [0]), axis=0)]
    #         bottom = [np.concatenate(([0], self.get_boundary(Orientation.bottom, field), [0]), axis=0)]
    #     else:
    #         top = [self.get_boundary(Orientation.top, field)]
    #         bottom = [self.get_boundary(Orientation.bottom, field)]
    #     matrix = np.concatenate((bottom, matrix, top), axis=0)
    #     return matrix

