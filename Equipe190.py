import numpy as np
import matplotlib.pyplot as plt
from time import time as time
from reglinA import reglinA
from reglinB import reglinB
from newton import newton
from regfreqA import regfreqA
from regfreqB import regfreqB

X = np.loadtxt("points.txt")
plt.scatter(X[:,0],X[:,1],s=3)
