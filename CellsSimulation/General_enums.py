import enum


class Field(enum.Enum):
    u = enum.auto()
    v = enum.auto()
    p = enum.auto()
    all = enum.auto()


class BoundaryConditionsType(enum.Enum):
    dirichlet = enum.auto()
    neumann = enum.auto()


class Orientation(enum.Enum):
    left = enum.auto()
    right = enum.auto()
    bottom = enum.auto()
    top = enum.auto()
    all = enum.auto()


class Information(enum.Flag):
    none = enum.auto()
    check_divergent = enum.auto()
    check_gradient_p_dot_u_vector = enum.auto()
    check_num_3 = enum.auto()
    all = check_divergent | check_gradient_p_dot_u_vector | check_num_3