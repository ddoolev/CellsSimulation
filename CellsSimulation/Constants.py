import simpy
# The distance between 2 coordinates on the greed
# The higher it is, the more accurate the simulation, and the longer it will take
RESOLUTION = 1

Re = 7500  # Reynolds number
# self.__redo_operators_matrix_boundaries()
DELTA_T = 0.001
GRID_SIZE = 20
TEST_NAME = "lead_driven_cavity"

ENV = simpy.Environment()
