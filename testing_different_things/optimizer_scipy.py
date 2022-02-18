import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
def cost_function(x, y):
    return np.cos(x) + abs(x)*0.1 - np.arctan(x**2-x**3) + y

x = np.arange(-10, 10, 0.1)
plt.plot(x, cost_function(x, 3))
plt.show()


min = scipy.optimize.minimize(cost_function,
                        np.array([10]),
                        args=(3,),
                        method='BFGS')