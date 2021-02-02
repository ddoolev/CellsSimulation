import numpy as np
import GeneralEnums as E


class _Grid:

    @staticmethod
    def _create_half_grid(delta):
        delta_half_grid = (_Grid._remove_boundaries(delta, E.Orientation.left) +
                           _Grid._remove_boundaries(delta, E.Orientation.right)) / 2
        delta_half_grid = np.concatenate(([delta[0] / 2], delta_half_grid, [delta[-1] / 2]))
        return delta_half_grid

    @staticmethod
    def _remove_boundaries(array, orientation):
        remove_boundary = {
            E.Orientation.left: _Grid._remove_boundaries_left(array),
            E.Orientation.right: _Grid._remove_boundaries_right(array),
            E.Orientation.all: _Grid._remove_boundaries_all(array)
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


class Grid2(_Grid):

    def __init__(self, delta_x, delta_y):
        self._delta_x = delta_x
        self._delta_y = delta_y
        self._delta_half_x = self._create_half_grid(delta_x)
        self._delta_half_y = self._create_half_grid(delta_y)

        self._x = np.concatenate(([0], np.cumsum(delta_x)))
        self._y = np.concatenate(([0], np.cumsum(delta_x)))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def delta_x(self):
        return self._delta_x

    @property
    def delta_y(self):
        return self._delta_y

    @property
    def delta_half_x(self):
        return self._delta_half_x

    @property
    def delta_half_y(self):
        return self._delta_half_y

    @property
    def delta_x_no_boundaries(self):
        return Grid2._remove_boundaries(self._delta_x, E.Orientation.all)

    @property
    def delta_y_no_boundaries(self):
        return Grid2._remove_boundaries(self._delta_y, E.Orientation.all)

    @property
    def delta_half_x_no_boundaries(self):
        return Grid2._remove_boundaries(self._delta_half_x, E.Orientation.all)

    @property
    def delta_half_y_no_boundaries(self):
        return Grid2._remove_boundaries(self._delta_half_y, E.Orientation.all)


class Grid3(Grid2):

    def __init__(self, delta_x, delta_y, delta_z):
        super().__init__(delta_x, delta_y)
        self._delta_z = delta_z
        self._delta_half_z = Grid3._create_half_grid(delta_z)

        self._z = np.concatenate(([0], np.cumsum(delta_z)))

    @property
    def z(self):
        return self._z

    @property
    def delta_z(self):
        return self._delta_z

    @property
    def delta_half_z(self):
        return self._delta_half_z

    @property
    def delta_z_no_boundaries(self):
        return Grid2._remove_boundaries(self._delta_z, E.Orientation.all)

    @property
    def delta_half_z_no_boundaries(self):
        return Grid2._remove_boundaries(self._delta_half_z, E.Orientation.all)
