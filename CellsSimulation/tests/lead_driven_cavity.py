import numpy as np
from Boundaries import Boundaries
from NavierStokesEquations import NavierStokesEquations, Information
import matplotlib.pyplot as plt
from general_enums import Field, Orientation
from Deltas import Delta2


def is_in_steady_state():
    epsilon = 0.00001
    delta_u = new_fields_matrix[Field.u] - old_fields_matrix[Field.u]
    delta_v = new_fields_matrix[Field.v] - old_fields_matrix[Field.v]
    delta_p = new_fields_matrix[Field.p] - old_fields_matrix[Field.p]
    if (delta_u.max() < epsilon) and (delta_v.max() < epsilon) and (delta_p.max() < epsilon):
        print("delta_u.max = ", delta_u.max())
        print("delta_v.max = ", delta_v.max())
        print("delta_p.max = ", delta_p.max())
        return True
    return False


if __name__ == "__main__":

    # change constants
    grid_size = 20
    time = 100000
    delta_t = 0.1

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

    domain = NavierStokesEquations(fields_matrix, delta_xy, boundaries, information=Information.all)

    domain.quiver()
    old_fields_matrix = domain.fields_matrix.copy()
    plt.pause(0.1)
    for i in range(time):
        plt.cla()
        domain.next_step()
        new_fields_matrix = domain.fields_matrix.copy()
        if is_in_steady_state():
            break
        old_fields_matrix = new_fields_matrix.copy()
        domain.quiver()
        plt.pause(0.1)
