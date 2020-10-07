import numpy as np
from general_enums import Orientation


class _Delta:

    @staticmethod
    def _create_half_grid(delta):
        delta_half_grid = (_Delta._remove_boundaries(delta, Orientation.left) +
                           _Delta._remove_boundaries(delta, Orientation.right)) / 2
        delta_half_grid = np.concatenate(([delta / 2], delta_half_grid, [delta / 2]))
        return delta_half_grid

    @staticmethod
    def _remove_boundaries(array, orientation):
        remove_boundary = {
            Orientation.left: _Delta._remove_boundaries_left(array),
            Orientation.right: _Delta._remove_boundaries_right(array),
            Orientation.all: _Delta._remove_boundaries_all(array)
        }
        return remove_boundary.get(orientation)

    @staticmethod
    def _remove_boundaries_all(array):
        return array[1:-1]

    @staticmethod
    def _remove_boundaries_left(array):
        return array[1:]

    @staticmethod
    def _remove_boundaries_right(array):
        return array[:-1]


class Delta2(_Delta):

    def __init__(self, delta_x, delta_y, delta_t):
        self._x = delta_x
        self._y = delta_y
        self._delta_t = delta_t
        self._half_x = self._create_half_grid(delta_x)
        self._half_y = self._create_half_grid(delta_y)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def half_x(self):
        return self._half_x

    @property
    def half_y(self):
        return self._half_y

    @property
    def x_no_boundaries(self):
        return Delta2._remove_boundaries(self._x, Orientation.all)

    @property
    def y_no_boundaries(self):
        return Delta2._remove_boundaries(self._y, Orientation.all)

    @property
    def half_x_no_boundaries(self):
        return Delta2._remove_boundaries(self._half_x, Orientation.all)

    @property
    def half_y_no_boundaries(self):
        return Delta2._remove_boundaries(self._half_y, Orientation.all)

    @property
    def t(self):
        return self._delta_t

    @t.setter
    def t(self, value):
        self._delta_t = value


class Delta3(Delta2):

    def __init__(self, delta_x, delta_y, delta_z):
        super().__init__(delta_x, delta_y)
        self._z = delta_z
        self._half_z = Delta3._create_half_grid(delta_z)

    @property
    def z(self):
        return self._z

    @property
    def half_z(self):
        return self._half_z

    @property
    def z_no_boundaries(self):
        return Delta2._remove_boundaries(self._z, Orientation.all)

    @property
    def half_z_no_boundaries(self):
        return Delta2._remove_boundaries(self._half_z, Orientation.all)
