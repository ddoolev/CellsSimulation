import numpy as np
import math
import Constants as C
import Cell
import random


class DefaultCell(Cell.Cell):
    #growth_time_line = np.array([])
    
    def __init__(self, center, r = 100, growthRate = 1, r_split = 200):
        self._center = center
        self._r = r
        self.calculateBoundries()
        self._r_split = r_split

        Cell.Cell.__init__(
            self, 
            Cell.CreateType.BOUNDARIES, 
            self._boundries,
            growthRate
        )

    def calculateBoundries(self):
        # initialization
        r = self._r
        center = self._center
        boundries = np.array([[r + self._center[0]],[center[1]]])

        # min angle on the circle between 2 adjacent points
        angle = math.acos((2*math.pow(r,2) - math.pow(C.RESOLUTION,2))/(2*math.pow(r,2)))        
        # num of points to create on the circle
        num_of_points = int(2*math.pi/angle)         
        # final angle between points
        angle = 2*math.pi/num_of_points
        current_angle = angle
        for i in range(num_of_points):
            [x,y] = self.Polar2Cartesian(r,current_angle)
            boundries = np.concatenate((boundries,[[x],[y]]),1)
            current_angle += angle

        self._boundries = boundries

    ###################################### Growth functions

    def _grow(self):
        self._r += self.growth_rate

    def _shouldGrow(self):
        return True

    ###################################### Split functions

    def _split(self, split_angle = -1, growth_rate = math.inf):
        self.removed = True
        if (split_angle == -1):
            split_angle = random.uniform(0, 180)
        if (growth_rate == math.inf):
            growth_rate = self.growth_rate
        new_r = self._r/2
        new_center1 = np.add(self._center, self.Polar2Cartesian(new_r, split_angle))
        new_center2 = np.subtract(self._center, self.Polar2Cartesian(new_r, split_angle))
        cell1 = DefaultCell(new_center1, new_r, growth_rate, self._r_split)
        cell2 = DefaultCell(new_center2, new_r, growth_rate, self._r_split)
        return [cell1, cell2]

    def setSplitSize(self, _r_split):
        self._r_split = _r_split

    def _shouldSplit(self):
        return (self._r == self._r_split)

    ###################################### Update functions

    def updateCell(self):
        return_statment = {}
        if (self._shouldGrow()):
            self._grow()
            if (self._shouldSplit()):
                return self._updateSplit()
            else:
                return self._updateGrow()


    def _updateSplit(self):
        new_cells = self._split()
        state = Cell.State.FINISHED_SPLITING
        general_status = {"state":state, "new_cells":new_cells}
        return general_status

    def _updateGrow(self):
        state = Cell.State.GROWING
        general_status = {"state":state}
        return general_status

    ###################################### Other functions

    def Polar2Cartesian(self,r,angle):
        x = r*math.cos(angle) + self._center[0]
        y = r*math.sin(angle) + self._center[1]
        return [x,y]


