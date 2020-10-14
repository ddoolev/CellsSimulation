import enum


class Field(enum.Enum):
    u = enum.auto()
    v = enum.auto()
    p = enum.auto()
    all = enum.auto()


class Orientation(enum.Enum):
    left = enum.auto()
    right = enum.auto()
    bottom = enum.auto()
    top = enum.auto()
    all = enum.auto()


class BoundaryConditionsType(enum.Enum):
    dirichlet = enum.auto()
    neumann = enum.auto()
