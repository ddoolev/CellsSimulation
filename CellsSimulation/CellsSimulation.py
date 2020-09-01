import Simulation as sim
from cell_creator.DefaultCell import DefaultCell as DCell

simulation = sim.Simulation()
round_cell = DCell(center=[0, 0])
simulation.add_cell(round_cell)
simulation.simulation_start()
