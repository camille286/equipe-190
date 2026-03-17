import numpy as np
import matplotlib.pyplot as plt
from time import time as time
from reglinA import reglinA
from reglinB import reglinB
from newton import newton

X = np.loadtxt("points.txt")
plt.scatter(X[:,0], X[:,1], s=3, c='plum')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Nuage de points")

plt.grid()
plt.show()

X = np.loadtxt("points.txt")

betaA = reglinA(X)
betaB = reglinB(X)

x = np.linspace(1, 6)
yA = betaA[0] + betaA[1]*x
yB = betaB[0] + betaB[1]*x

plt.scatter(X[:,0], X[:,1], s=3, label="Points", c='plum')
plt.plot(x, yA, label="reglinA")
plt.plot(x, yB, label="reglinB")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Régression linéaire")
plt.legend()
plt.grid()
plt.show()