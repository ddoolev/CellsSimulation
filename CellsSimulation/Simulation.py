import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import Constants as C
import Cell


class Simulation:

    def __init__(self, size=1000, cells=np.array([]), time=20):
        self.size = size
        self.cells = cells
        self.time = time
        self.fig = plb.figure()
        self.axes = plb.axes(xlim=[-self.size, self.size], ylim=[-self.size, self.size])
        self.__boundaries = [plb.plot([], [])[0] for _ in range(len(self.cells))]

    def add_cell(self, cell):
        self.add_cells([cell])

    def add_cells(self, cells):
        self.cells = np.concatenate((self.cells, cells))

    def remove_cell(self, cell_to_remove):
        cell_index = np.where(self.cells == cell_to_remove)
        if len(cell_index) != 0:
            self.cells = np.delete(self.cells, cell_index[0])
            return True
        return False

    def simulation_start(self):
        # initial data
        C.ENV.process(self.main_loop())
        for cell in self.cells:
            C.ENV.process(cell.update())
        C.ENV.run(until=self.time)

    def main_loop(self):
        self.axes.set_aspect('equal')
        # make the main loop run after all the cells grow
        yield C.ENV.timeout(C.DELTA_T / 1000)
        while True:
            plt.cla()
            for cell in self.cells:
                cell_status = cell.get_status()
                if cell_status["state"] == Cell.State.FINISHED_SPLITTING:
                    self.remove_cell(cell)
                    self.add_cells(cell_status["new_cells"])
            self.calculate_boundaries()
            boundaries = self.get_boundaries()
            for cell_boundary in boundaries:
                self.axes.add_line(cell_boundary)

            plt.pause(0.1)
            yield C.ENV.timeout(C.DELTA_T)

    def calculate_boundaries(self):
        for cell in self.cells:
            cell.calculate_boundaries()

    def get_boundaries(self):
        boundaries = [plb.plot([], [])[0] for _ in range(len(self.cells))]
        for i, cell in enumerate(self.cells, start=0):
            cell_boundaries = cell.get_boundaries()
            boundaries[i].set_data(cell_boundaries[0], cell_boundaries[1])
        return boundaries
