from matplotlib import pyplot as plt
import General_enums as E
import numpy as np
from NavierStokesEquations import NavierStokesEquations
from matplotlib import animation


class Simulation:

    def __init__(self, fields, grid, boundaries, delta_t=0.01, time_steps=100,
                 boundary_conditions_type=E.BoundaryConditionsType.neumann,
                 information=E.Information.none,
                 cells=np.array([])):
        self.__domain = NavierStokesEquations(fields, grid, boundaries, delta_t,
                                              boundary_conditions_type, information)
        self.__time_steps = time_steps
        self.__cells = cells
        self.__ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
        self.__fig = plt.figure()
        self.__delta_t = delta_t

    def __animation_init(self):
        U = np.cos(X + num * 0.1)
        V = np.sin(Y + num * 0.1)

        Q.set_UVC(U, V)

        return Q,


        return self.__domain.quiver()

    def __animation_main(self, i):
        self.__domain.next_step()
        return self.__domain.quiver()

    def start(self):
        anim = animation.FuncAnimation(self.__fig, self.__animation_main, init_func=self.__animation_init,
                                       frames=self.__time_steps, interval=20, blit=True)
        # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        plt.show()