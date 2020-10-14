import numpy as np
from Boundaries import Boundaries
from NavierStokesEquations import NavierStokesEquations, Information
import matplotlib.pyplot as plt
from general_enums import Field, Orientation
from Deltas import Delta2
import Constants as C


def is_in_steady_state():
    return (delta_u.max() < epsilon) and (delta_v.max() < epsilon) and (delta_p.max() < epsilon)


if __name__ == "__main__":
    # change constants
    grid_size = C.GRID_SIZE - 1

    u_matrix = np.full((grid_size + 1, grid_size), 0)
    v_matrix = np.full((grid_size, grid_size + 1), 0)
    p_matrix = np.full((grid_size + 1, grid_size + 1), 0)
    fields_matrix = {Field.u: u_matrix, Field.v: v_matrix, Field.p: p_matrix}

    delta = np.full(grid_size + 1, 1/(grid_size + 1))
    delta_xy = Delta2(delta, delta, C.DELTA_T)

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

    domain = NavierStokesEquations(fields_matrix, delta_xy, boundaries, information=Information.check_divergent)

    new_fields_matrix = domain.fields_matrix
    time = 0
    step_counter = 0
    epsilon = 0.00001
    while True:
        print("time = ", time)
        old_fields_matrix = domain.fields_matrix.copy()
        plt.cla()
        domain.next_step()
        # domain.quiver()
        # plt.pause(0.01)
        if step_counter % 10 == 0:
            delta_u = new_fields_matrix[Field.u] - old_fields_matrix[Field.u] / new_fields_matrix[Field.u]
            delta_v = new_fields_matrix[Field.v] - old_fields_matrix[Field.v] / new_fields_matrix[Field.v]
            delta_p = new_fields_matrix[Field.p] - old_fields_matrix[Field.p] / new_fields_matrix[Field.p]
            middle_u = new_fields_matrix[Field.u].T[int((grid_size + 2)/2)].T
            middle_v = new_fields_matrix[Field.v][int((grid_size + 2)/2)]
            print("middle u = ", middle_u)
            print("middle v = ", middle_v)
            if is_in_steady_state():
                print("In steady state")
                break
        step_counter += 1
        time += C.DELTA_T
        print()
