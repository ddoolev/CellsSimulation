import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as pyplot
import Constants as C
import Cell


class Simulation:

    def __init__(self, size=1000, cells=np.array([]), time=20):
        self.size = size
        self.cells = cells
        self.time = time
        self.fig = pylab.figure()
        self.axes = pylab.axes(xlim=[-self.size, self.size], ylim=[-self.size, self.size])
        self.__boundaries = [pylab.plot([], [])[0] for _ in range(len(self.cells))]

    def addCell(self, cell):
        self.addCells([cell])

    def addCells(self, cells):
        self.cells = np.concatenate((self.cells, cells))

    def removeCell(self, cell_to_remove):
        cell_index = np.where(self.cells == cell_to_remove)
        if len(cell_index) != 0:
            self.cells = np.delete(self.cells, cell_index[0])
            return True
        return False

    def simulationStart(self):
        # initial data  
        C.ENV.process(self.mainLoop())
        for cell in self.cells:
            C.ENV.process(cell.update())
        C.ENV.run(until=self.time)

    def mainLoop(self):
        self.axes.set_aspect('equal')
        # make the main loop run after all the cells grow
        yield C.ENV.timeout(C.TIME_STEP / 1000)
        while True:
            pyplot.cla()
            for cell in self.cells:
                cell_status = cell.getStatus()
                if cell_status["state"] == Cell.State.FINISHED_SPLITING:
                    self.removeCell(cell)
                    self.addCells(cell_status["new_cells"])
            self.calculateBoundaries()
            boundaries = self.getBoundaries()
            for cell_boundary in boundaries:
                self.axes.add_line(cell_boundary)

            pyplot.pause(0.1)
            yield C.ENV.timeout(C.TIME_STEP)

    def calculateBoundaries(self):
        for cell in self.cells:
            cell.calculateBoundaries()

    def getBoundaries(self):
        boundaries = [pylab.plot([], [])[0] for _ in range(len(self.cells))]
        for i, cell in enumerate(self.cells, start=0):
            cell_boundaries = cell.getBoundaries()
            boundaries[i].set_data(cell_boundaries[0], cell_boundaries[1])
        return boundaries
