import simpy

RESOLUTION = 1 # The distance between 2 coordinates on the greed
                # The higher it is, the more accurate the simulation, and the longer it will take
Re = 1 # Reynolds number
TIME_STEP = 1

ENV = simpy.Environment() 