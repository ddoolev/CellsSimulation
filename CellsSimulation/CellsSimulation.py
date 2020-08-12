import Simulation as sim
from cell_types.DefaultCell import DefaultCell as DCell

simulation = sim.Simulation()
round_cell = DCell(center = [0,0],r = 50, r_split = 100)
simulation.addCell(round_cell)
simulation.simulationStart()




#import matplotlib.pylab as plt
#import Simulation as sim

#x=[-1 ,0.5 ,1,-0.5]
#y=[ 0.5,  1, -0.5, -1]
#plt.plot(x,y)
#plt.axis('equal')
#plt.show()




#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#import pandas as pd
#import numpy as np

#N=11

#dataframes = [pd.DataFrame({"x":np.sort(np.random.rand(10)*100),
#                            "y1":np.random.rand(10)*30,
#                            "y2":np.random.rand(10)*30}) for _ in range(N)]

#fig = plt.figure()
#ax = plt.axes(xlim=(0,100), ylim=(0,30))

#lines = [plt.plot([], [])[0] for _ in range(2)]

#def animate(i):
#    lines[0].set_data(dataframes[i]["x"], dataframes[i]["y1"])
#    lines[1].set_data(dataframes[i]["x"], dataframes[i]["y2"])
#    return lines

#anim = animation.FuncAnimation(fig, animate, 
#           frames=N, interval=20, blit=True)

#plt.show()

