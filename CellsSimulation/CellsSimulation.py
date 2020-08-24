import numpy as np
import Simulation as sim
from cell_creator.DefaultCell import DefaultCell as DCell
from navier_stokes_equations.MathOperations import MathOperations

#simulation = sim.Simulation()
#round_cell = DCell(center = [0,0],r = 50, r_split = 100)
#simulation.addCell(round_cell)
#simulation.simulationStart()

array_size = 5
p_array = np.random.randint(10, size = (array_size,array_size))
delta_vector = np.arange(1,array_size)

print(p_array)
print(delta_vector)
print(MathOperations.gradientY(p_array,delta_vector))

