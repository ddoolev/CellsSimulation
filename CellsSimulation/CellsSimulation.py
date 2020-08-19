import numpy as np
import Simulation as sim
from cell_creator.DefaultCell import DefaultCell as DCell
from scipy.sparse import spdiags
from LaplaceOperator import LaplaceOperator

#simulation = sim.Simulation()
#round_cell = DCell(center = [0,0],r = 50, r_split = 100)
#simulation.addCell(round_cell)
#simulation.simulationStart()


array_size = 1000
data_array = np.random.randint(10, size=(array_size, array_size))
#print(data_array)

delta_vector = np.full((array_size-1), 1)

laplas = LaplaceOperator(delta_vector, delta_vector)
results = laplas.laplacianOperation(data_array)

print(results)



