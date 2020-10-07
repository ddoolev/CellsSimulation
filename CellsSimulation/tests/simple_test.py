import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


matrix = [[1,2],[3,4],[5,6]]
matrix = sp.csr_matrix(matrix)

print(matrix.get_shape == 2)

plt.show()
