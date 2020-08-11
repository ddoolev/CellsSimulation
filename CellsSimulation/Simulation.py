import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation
import Constants as C
import Cell


class Simulation:
    TIME_STEP = 1

    def __init__(self, size = 1000, cells = np.array([]), time = 2):
        self.size = size
        self.cells = cells
        self.time = time

    def addCell(self, cell):
        self.addCells([cell])

    def addCells(self, cells):
        self.cells = np.append(self.cells, cells)
            
    def removeCell(self, cell_remove):
        cell_index = np.where(self.cells == cell_remove)
        if (cell_index.size != 0):
            cells = np.delete(cells, cell_index)
            return True
        else:
            return False

    def simulationStart(self):
        # initial data  
        self.fig = plt.figure()
        self.axes = plt.axes(xlim=[-self.size,self.size], ylim=[-self.size,self.size])
        self.axes.set_aspect('equal')

        ani = animation.FuncAnimation(self.fig, self.simulationMainLoop, 
                                      init_func=self.simulationInit, 
                                      frames=self.time, interval=20,
                                      blit=True)
        plt.show()

    def simulationInit(self):
        self.__boundries = [plt.plot([], [])[0] for _ in range(len(self.cells))]
        return self.__boundries

    def simulationMainLoop(self, i):
        for cell in self.cells:
            general_status = cell.updateCell()
            if (general_status["state"] == Cell.State.FINISHED_SPLITING):
                self.removeCell(cell)
                self.addCells(general_status["new_cells"])

        self.calculateBoundries()
        self.getBoundries()
        return self.__boundries
        

    def calculateBoundries(self):
        for cell in self.cells:
            cell.calculateBoundries()

    def getBoundries(self):
        self.__boundries = [plt.plot([], [])[0] for _ in range(len(self.cells))]
        for i, cell in enumerate(self.cells, start=0):
            cellBoundries = cell.getBoundries()
            self.__boundries[i].set_data(cellBoundries[0], cellBoundries[1])
