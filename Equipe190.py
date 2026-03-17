import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt("points.txt")
plt.scatter(X[:,0], X[:,1], s=3, c='purple')

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Nuage de points")

plt.grid()
plt.show()
