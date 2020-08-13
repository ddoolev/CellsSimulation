import numpy as np
import Simulation as sim
from cell_creator.DefaultCell import DefaultCell as DCell
from scipy.sparse import spdiags

#simulation = sim.Simulation()
#round_cell = DCell(center = [0,0],r = 50, r_split = 100)
#simulation.addCell(round_cell)
#simulation.simulationStart()

data = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 4]])

diags = np.array([-2,-1,0])

print(spdiags(data, diags, 4, 4))


