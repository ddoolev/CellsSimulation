import enum
from sympy import *
from sympy.geometry import *


class CreateType(enum.Enum):
    BOUNDARIES = 0
# SOURCE


class State(enum.Enum):
    STATIC = 0
    GROWING = 1
    SPLITTING = 2
    FINISHED_SPLITING = 3
    DEAD = 4
    REMOVED = 5


class Cell:
    toRemoved = False
    general_status = {}

    def __init__(self,
                 cell_create_type=0,
                 points=[],
                 growth_rate=1):

        self._growth_rate = growth_rate  # how much to grow each time step
        if cell_create_type == CreateType.BOUNDARIES:
            self._boundaries = points
        state = State.GROWING
        self._general_status = {"state": state}

    def get_boundaries(self):
        return self._boundaries

    def remove(self):
        self.toRemoved = True

    def _grow(self):
        pass

    def update_cell(self):
        pass
