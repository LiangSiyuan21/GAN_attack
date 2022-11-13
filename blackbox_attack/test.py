import numpy as np
cross_point = np.random.rand(2, 3) < 0.5
m1 = np.random.rand(1, 2, 3)
m2 = np.random.rand(1, 2, 3)
temp = m2[0][cross_point]
m1[0, cross_point] = m2[0, cross_point]
print('test')