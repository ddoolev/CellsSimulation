import numpy as np
import Simulation as sim
from cell_creator.DefaultCell import DefaultCell as DCell
from scipy.sparse import spdiags
from LaplaceOperator import LaplaceOperator

#simulation = sim.Simulation()
#round_cell = DCell(center = [0,0],r = 50, r_split = 100)
#simulation.addCell(round_cell)
#simulation.simulationStart()

data_array = np.array([[1,2,3,4],
                       [5,6,7,8],
                       [9,10,11,12],
                       [13,14,15,16]])

delta_vector = np.array([1,1,1,1])

laplas = LaplaceOperator(delta_vector, delta_vector)
results = laplas.laplacianOperation(data_array)

print(results)



