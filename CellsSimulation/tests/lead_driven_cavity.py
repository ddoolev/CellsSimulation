import numpy as np
from Boundaries import Boundaries
from NavierStokesEquations import NavierStokesEquations
import matplotlib.pyplot as plt
from general_enums import Fields, Delta, Orientation

if __name__ == "__main__":

    grid_size = 20
    time = 100000

    u_matrix = np.full((grid_size + 1, grid_size), 0)
    v_matrix = np.full((grid_size, grid_size + 1), 0)
    p_matrix = np.full((grid_size + 1, grid_size + 1), 0)
    fields_matrix = {Fields.u: u_matrix, Fields.v: v_matrix, Fields.p: p_matrix}

    delta = np.full(grid_size + 1, 1)
    delta_matrix = {Delta.x: delta, Delta.y: delta}

    full_0 = np.full(grid_size + 1, 0)
    full_1 = np.full(grid_size + 1, 1)

    boundary_left = {Fields.u: full_0, Fields.v: full_0[1:], Fields.p: full_0}
    boundary_right = {Fields.u: full_0, Fields.v: full_0[1:], Fields.p: full_0}
    boundary_top = {Fields.u: full_1[1:], Fields.v: full_0, Fields.p: full_0}
    boundary_bottom = {Fields.u: full_0[1:], Fields.v: full_0, Fields.p: full_0}

    boundaries = Boundaries({Orientation.left: boundary_left,
                             Orientation.right: boundary_right,
                             Orientation.top: boundary_top,
                             Orientation.bottom: boundary_bottom})

    domain = NavierStokesEquations(fields_matrix, delta_matrix, boundaries)

    # plt.axes.set_aspect('equal')
    domain.quiver()
    plt.pause(0.1)
    for i in range(time):
        plt.cla()
        domain.next_step()
        domain.quiver()
        plt.pause(0.1)
