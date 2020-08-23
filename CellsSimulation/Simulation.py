import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as pyplot
from matplotlib import animation
import Constants as C
import Cell


class Simulation:

    def __init__(self, size = 1000, cells = np.array([]), time = 2):
        self.size = size
        self.cells = cells
        self.time = time

    def addCell(self, cell):
        self.addCells([cell])

    def addCells(self, cells):
        self.cells = np.concatenate((self.cells, cells))
            
    def removeCell(self, cell_to_remove):
        cell_index = np.where(self.cells == cell_to_remove)
        if (len(cell_index) != 0):
            self.cells = np.delete(self.cells, cell_index[0])
            return True
        return False

    def simulationStart(self):
        # initial data  
        self.fig = pylab.figure()
        self.axes = pylab.axes(xlim=[-self.size,self.size], ylim=[-self.size,self.size])
        self.axes.set_aspect('equal')

        while(True):
            pyplot.cla()
            for cell in self.cells:
                general_status = cell.updateCell()
                if (general_status["state"] == Cell.State.FINISHED_SPLITING):
                    self.removeCell(cell)
                    self.addCells(general_status["new_cells"])

            self.calculateBoundries()
            self.getBoundries()
            for cell_boundry in self.__boundries:
                self.axes.add_line(cell_boundry)
            pyplot.pause(0.0001)


    def calculateBoundries(self):
        for cell in self.cells:
            cell.calculateBoundries()

    def getBoundries(self):
        self.__boundries = [pylab.plot([], [])[0] for _ in range(len(self.cells))]
        for i, cell in enumerate(self.cells, start=0):
            cell_boundries = cell.getBoundries()
            self.__boundries[i].set_data(cell_boundries[0], cell_boundries[1])
