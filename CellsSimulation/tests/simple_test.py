import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# create some random data
ptx1 = [random.randint(0,100) for x in range(20)]
pty1 = [random.randint(0,100) for x in range(20)]

fig = plt.figure()
ax = fig.add_subplot(111)

def animate(i):
    # use i-th elements from data
    ax.scatter(ptx1[:i], pty1[:i], c='red')

    # or add only one element from list
    #ax.scatter(ptx1[i], pty1[i], c='red')

ani = FuncAnimation(fig, animate, frames=20, interval=500)

plt.show()