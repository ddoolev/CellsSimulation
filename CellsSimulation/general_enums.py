import enum


class Fields(enum.Enum):
    u = 0
    v = 1
    p = 2
    all = 3


class Delta(enum.Enum):
    x = 0
    y = 1


class Orientation(enum.Enum):
    left = 0
    right = 1
    bottom = 2
    top = 3
    all = 4