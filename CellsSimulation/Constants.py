import simpy
# The distance between 2 coordinates on the greed
# The higher it is, the more accurate the simulation, and the longer it will take
RESOLUTION = 1

Re = 0.001  # Reynolds number
DELTA_T = 0.001

ENV = simpy.Environment()
