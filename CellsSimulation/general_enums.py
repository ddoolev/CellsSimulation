import enum


class Field(enum.Enum):
    u = 0
    v = 1
    p = 2
    all = 3


class Orientation(enum.Enum):
    left = 0
    right = 1
    bottom = 2
    top = 3
    all = 4


class BoundaryConditionsType(enum.Enum):
    dirichlet = 0
    neumann = 1