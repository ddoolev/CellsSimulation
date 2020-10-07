import numpy as np
from Boundaries import Boundaries
from NavierStokesEquations import NavierStokesEquations
import matplotlib.pyplot as plt
from general_enums import Field, Orientation
from Deltas import Delta2

if __name__ == "__main__":

    # change constants
    grid_size = 10
    time = 100000
    delta_t = 0.01

    u_matrix = np.full((grid_size + 1, grid_size), 0)
    v_matrix = np.full((grid_size, grid_size + 1), 0)
    p_matrix = np.full((grid_size + 1, grid_size + 1), 0)
    fields_matrix = {Field.u: u_matrix, Field.v: v_matrix, Field.p: p_matrix}

    delta = np.full(grid_size + 1, 1/(grid_size + 1))
    delta_xy = Delta2(delta, delta, delta_t)

    full_0 = np.full(grid_size + 1, 0)
    full_1 = np.full(grid_size + 1, 1)

    boundary_left = {Field.u: full_0, Field.v: full_0[1:], Field.p: full_0}
    boundary_right = {Field.u: full_0, Field.v: full_0[1:], Field.p: full_0}
    boundary_top = {Field.u: full_1[1:], Field.v: full_0, Field.p: full_0}
    boundary_bottom = {Field.u: full_0[1:], Field.v: full_0, Field.p: full_0}

    boundaries = Boundaries({Orientation.left: boundary_left,
                             Orientation.right: boundary_right,
                             Orientation.top: boundary_top,
                             Orientation.bottom: boundary_bottom})

    domain = NavierStokesEquations(fields_matrix, delta_xy, boundaries)

    domain.quiver()
    plt.pause(0.5)
    for i in range(time):
        plt.cla()
        domain.next_step()
        domain.quiver()
        plt.pause(0.5)
