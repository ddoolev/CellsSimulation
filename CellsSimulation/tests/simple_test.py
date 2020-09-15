import numpy as np
import matplotlib.pyplot as plt

u_matrix = np.full((2, 20), 0)
v_matrix = np.full((2, 20), 0)

x = [i for i in range(2)]
y = [i for i in range(20)]

xx, yy = np.meshgrid(x, y)

plt.quiver(xx, yy, u_matrix, v_matrix)
plt.show()
