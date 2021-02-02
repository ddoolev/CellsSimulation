import numpy as np
from Boundaries import Boundaries
from NavierStokesEquations import NavierStokesEquations
import matplotlib.pyplot as plt
from General_enums import Field, Orientation, Information
from Grid import Grid2
from Fields import Fields2
import Constants as C


def is_in_steady_state():
    return (delta_u.max() < epsilon) and (delta_v.max() < epsilon) and (delta_p.max() < epsilon)


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

    domain = NavierStokesEquations(fields_matrix, delta_xy, boundaries, delta_t, information=Information.all)

    new_fields = domain.fields
    time = 0
    step_counter = 0
    epsilon = 0.00001
    while True:
        print("time = ", time)
        old_fields = domain.fields.copy()
        plt.cla()
        domain.next_step()
        delta_u = (new_fields.u - old_fields.u) / new_fields.u
        delta_v = (new_fields.v - old_fields.v) / new_fields.v
        tmp_filed_matrix_p = new_fields.p
        tmp_filed_matrix_p[0][0] = 1
        delta_p = (new_fields.p - old_fields.p) / tmp_filed_matrix_p
        print("delta_u max = ", delta_u.max(), "\tdelta_v max = ", delta_v.max(), "\tdelta_p max = ", delta_p.max())
        if step_counter % 10 == 0:
            plt.title(("time = ", str(time)))
            domain.quiver()
            plt.pause(0.1)
            if is_in_steady_state():
                middle_u = new_fields.u.T[int((grid_size + 2) / 2)].T
                middle_v = new_fields.v[int((grid_size + 2) / 2)]
                print("middle u = ", middle_u)
                print("middle v = ", middle_v)
                print("In steady state")
                print("Reynold number = ", C.RE)
                plt.title(("time = ", str(time), "\tReynold number = ", C.RE))
                domain.quiver()
                plt.show()
                break
        step_counter += 1
        time += C.DELTA_T
        print()
        print()
