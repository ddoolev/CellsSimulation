import numpy as np
import math
import Constants as C
import Cell
import random


class DefaultCell(Cell.Cell):

    def __init__(self, center, r=10.0, growth_rate=1, r_split=20):
        self._center = center
        self._r = r
        self.calculate_boundaries()
        self._r_split = r_split
        self._boundaries = np.array([])

        Cell.Cell.__init__(
            self,
            Cell.CreateType.BOUNDARIES,
            self._boundaries,
            growth_rate
        )

    def calculate_boundaries(self):
        # initialization
        r = self._r
        center = self._center
        boundaries = np.array([[r + self._center[0]], [center[1]]])

        # min angle on the circle between 2 adjacent points
        angle = math.acos((2 * math.pow(r, 2) - math.pow(C.RESOLUTION, 2)) / (2 * math.pow(r, 2)))
        # num of points to create on the circle
        num_of_points = int(2 * math.pi / angle)
        # final angle between points
        angle = 2 * math.pi / num_of_points
        current_angle = angle
        for i in range(num_of_points):
            [x, y] = self.polar_to_cartesian(r, current_angle)
            boundaries = np.concatenate((boundaries, [[x], [y]]), 1)
            current_angle += angle

        self._boundaries = boundaries

    ###################################### Growth functions

    def _grow(self):
        self._r += self._growth_rate

    def _should_grow(self):
        return True

    ###################################### Split functions

    def _split(self, split_angle=-1, growth_rate=math.inf):
        self.removed = True
        if split_angle == -1:
            split_angle = random.uniform(0, 180)
        if growth_rate == math.inf:
            growth_rate = self._growth_rate
        new_r = self._r / 2
        new_center1 = np.add(self._center, self.polar_to_cartesian(new_r, split_angle))
        new_center2 = np.subtract(self._center, self.polar_to_cartesian(new_r, split_angle))
        cell1 = DefaultCell(new_center1, new_r, growth_rate, self._r_split)
        cell2 = DefaultCell(new_center2, new_r, growth_rate, self._r_split)
        return [cell1, cell2]

    def set_split_size(self, _r_split):
        self._r_split = _r_split

    def _should_split(self):
        return self._r == self._r_split

    ###################################### Update functions

    def update(self):
        while True:
            yield C.ENV.timeout(C.TIME_STEP)
            if self._should_grow():
                self._grow()
                if self._should_split():
                    self._update_split()
                else:
                    self._update_grow()

    def _update_split(self):
        new_cells = self._split()
        for cell in new_cells:
            C.ENV.process(cell.update())
        state = Cell.State.FINISHED_SPLITING
        self._general_status = {"state": state, "new_cells": new_cells}

    def _update_grow(self):
        state = Cell.State.GROWING
        self._general_status = {"state": state}

    def get_status(self):
        return self._general_status

    ###################################### Other functions

    def polar_to_cartesian(self, r, angle):
        x = r * math.cos(angle) + self._center[0]
        y = r * math.sin(angle) + self._center[1]
        return [x, y]
