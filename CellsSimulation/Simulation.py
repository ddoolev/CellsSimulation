from matplotlib import pyplot as plt
import GeneralEnums as E
import numpy as np
from NavierStokesEquations import NavierStokesEquations
from matplotlib import animation
from MathFunctions import interpolate
import Point
from Boundaries import Boundaries


class Simulation:

    def __init__(self, fields, grid, boundaries, delta_t=0.01, time_steps=100,
                 boundary_conditions_type=E.BoundaryConditionsType.neumann,
                 information=E.Information.none,
                 cells=np.array([])):
        self.__num_of_points = 10
        self.__domain = NavierStokesEquations(fields, grid, boundaries, delta_t,
                                              boundary_conditions_type, information)
        self.__grid = grid
        self.__boundaries = boundaries
        self.__time_steps = time_steps
        self.__cells = cells
        self.__fig = plt.figure()
        self.__ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
        self.__delta_t = delta_t

        self.__points = Point.create_random_point_array(self.__num_of_points)
        array_of_points = self.__points.to_array()
        self.__points_scatter = self.__ax.scatter(array_of_points[0], array_of_points[1], s=60)
        self.__lines = [[[], []]]
        self.__lines2d = self.__ax.plot([], [], lw=1)
        for i in range(self.__num_of_points - 1):
            self.__lines.append([[], []])
            self.__lines2d = np.concatenate((self.__lines2d, self.__ax.plot([], [], lw=1)))

    def __animation_init(self):
        self.__update_lines_and_points()
        return self.__lines2d

    def __animation_main(self, i):
        print("*********************** time step " + str(i))
        self.__domain.next_step()
        u_field = Boundaries.add_boundaries(self.__domain.fields.u, self.__boundaries, E.Field.u, E.Orientation.all)
        v_field = Boundaries.add_boundaries(self.__domain.fields.v, self.__boundaries, E.Field.v, E.Orientation.all)
        grid_delta_y = np.concatenate(([self.__grid.delta_y[0]/2], self.__grid.delta_y, [self.__grid.delta_y[-1]/2]))
        grid_delta_x = np.concatenate(([self.__grid.delta_x[0]/2], self.__grid.delta_x, [self.__grid.delta_x[-1]/2]))
        u_field = interpolate(u_field, self.__grid.x, self.__grid.half_y,
                              self.__grid.delta_half_x, grid_delta_y, self.__points)
        v_field = interpolate(v_field, self.__grid.half_x, self.__grid.y,
                              grid_delta_x, self.__grid.delta_half_y, self.__points)
        self.__points.x = self.__points.x + u_field * self.__delta_t
        self.__points.y = self.__points.y + v_field * self.__delta_t
        self.__update_lines_and_points()
        return self.__lines2d, self.__points.to_array_of_points()

    def __update_lines_and_points(self):
        array_of_points = self.__points.to_array_of_points()
        for i in range(self.__num_of_points):
            self.__lines[i][0].append(array_of_points[i][0])
            self.__lines[i][1].append(array_of_points[i][1])
            self.__lines2d[i].set_data(self.__lines[i][0], self.__lines[i][1])
        array_of_points = self.__points.to_array()
        self.__points_scatter = self.__ax.scatter(array_of_points[0], array_of_points[1], s=60)

    def start(self):
        self.__anim = animation.FuncAnimation(self.__fig, self.__animation_main, init_func=self.__animation_init,
                                       frames=self.__time_steps, interval=1, blit=True)
        self.__anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        plt.show()
