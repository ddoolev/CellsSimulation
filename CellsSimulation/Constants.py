import simpy
# The distance between 2 coordinates on the greed
# The higher it is, the more accurate the simulation, and the longer it will take
RESOLUTION = 1

RE = 100  # Reynolds number
# self.__redo_operators_matrix_boundaries()
DELTA_T = 0.01
GRID_SIZE = 50
TEST_NAME = "lead_driven_cavity"



ENV = simpy.Environment()
