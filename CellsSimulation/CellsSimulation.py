import numpy as np
import Simulation as sim
from cell_creator.DefaultCell import DefaultCell as DCell
from navier_stokes_equations.MathOperations import MathOperations

simulation = sim.Simulation()
round_cell = DCell(center = [0,0])
simulation.addCell(round_cell)
simulation.simulationStart()

