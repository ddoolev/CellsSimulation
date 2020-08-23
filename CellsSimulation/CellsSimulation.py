import numpy as np
import Simulation as sim
from cell_creator.DefaultCell import DefaultCell as DCell
from scipy.sparse import spdiags
from LaplaceOperator import LaplaceOperator

simulation = sim.Simulation()
round_cell = DCell(center = [0,0],r = 10, r_split = 20)
simulation.addCell(round_cell)
simulation.simulationStart()



