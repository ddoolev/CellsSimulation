from time import sleep

import numpy as np
from Boundaries import Boundaries
from GeneralEnums import Field, Orientation, Information
from Grid import Grid2
from Fields import Fields2
import Constants as C
from Simulation import Simulation


if __name__ == "__main__":
    # change constants
    grid_size = C.GRID_SIZE - 1
    delta_t = 0.001

    u_matrix = np.full((grid_size + 1, grid_size), 0)
    v_matrix = np.full((grid_size, grid_size + 1), 0)
    p_matrix = np.full((grid_size + 1, grid_size + 1), 0)
    fields_matrix = Fields2(v_matrix, u_matrix, p_matrix)

    delta = np.full(grid_size + 1, 1/(grid_size + 1))
    delta_xy = Grid2(delta, delta)

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

    sim = Simulation(fields_matrix, delta_xy, boundaries, delta_t, information=Information.all)
    sim.start()
    sleep(1000000)

