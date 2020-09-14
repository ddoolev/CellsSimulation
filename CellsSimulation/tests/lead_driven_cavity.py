import numpy as np
from Boundaries import Boundaries
from NavierStokesEquations import NavierStokesEquations
import matplotlib.pyplot as plt
from enums import Fields

if __name__ == "__main__":

    simulation_size = 20
    time = 100000

    u_matrix = np.full((simulation_size, simulation_size-1), 0)
    v_matrix = np.full((simulation_size-1, simulation_size), 0)
    p_matrix = np.full((simulation_size, simulation_size), 0)

    delta = np.full(simulation_size, 1)

    full_0 = np.full(simulation_size, 0)
    full_1 = np.full(simulation_size, 1)

    boundary_left = {Fields.U: full_0, Fields.V: full_0[1:], Fields.P: full_0}
    boundary_right = {Fields.U: full_0, Fields.V: full_0[1:], Fields.P: full_0}
    boundary_top = {Fields.U: full_1[1:], Fields.V: full_0, Fields.P: full_0}
    boundary_bottom = {Fields.U: full_0[1:], Fields.V: full_0, Fields.P: full_0}

    boundaries = Boundaries(boundary_left, boundary_right, boundary_top, boundary_bottom)

    domain = NavierStokesEquations(p_matrix, v_matrix, u_matrix, delta, delta, boundaries)

    # plt.axes.set_aspect('equal')
    domain.quiver()
    plt.pause(0.1)
    for i in range(time):
        plt.cla()
        domain.quiver()
        plt.pause(0.1)
