import numpy as np
import matplotlib.pyplot as plt
from time import time as time
from reglinA import reglinA
from reglinB import reglinB
from newton import newton
from regfreqA import regfreqA
from regfreqB import regfreqB

X = np.loadtxt("points.txt")
plt.scatter(X[:,0], X[:,1], s=3, c='plum')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Figure 1: Nuage de points")

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
plt.title("Figure 2: Régression linéaire")
plt.legend()
plt.grid()
plt.show()

# Charger les données
X = np.loadtxt("points.txt")

# Extraire x et y
x_data = X[:, 0]
y_data = X[:, 1]

# Définir F(beta) : doit retourner un vecteur colonne (1000,1)
F = lambda beta: (
    beta[0] + beta[1] * np.sqrt(x_data - beta[2]) - y_data
).reshape(-1, 1)

# Définir J(beta) : doit retourner une matrice (1000,3)
J = lambda beta: np.column_stack((
    np.ones(len(x_data)),
    np.sqrt(x_data - beta[2]),
    -beta[1] / (2 * np.sqrt(x_data - beta[2]))
))

# Point de départ
beta0 = np.array([[1.0], [1.0], [1.0]])

# Appel de Newton
beta = newton(beta0, F, J, 1e-7, 20)

print("beta =")
print(beta)

# Tracé de la courbe de régression
x = np.linspace(1, 6, 300)
y = beta[0,0] + beta[1,0] * np.sqrt(x - beta[2,0])

plt.figure()
plt.scatter(x_data, y_data, s=5, label="Points", c='plum')
plt.plot(x, y, label="Régression non linéaire", linewidth=2, c='red')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Figure 3: Régression non linéaire par Newton")
plt.legend()
plt.grid()
plt.show()

X = np.loadtxt("points.txt")

betaA5 = regfreqA(X, 5)
betaB5 = regfreqB(X, 5)

print("betaA5 =")
print(betaA5)

print("betaB5 =")
print(betaB5)

print("shape A5 =", betaA5.shape)
print("shape B5 =", betaB5.shape)

X = np.loadtxt("points.txt")
x_data = X[:, 0]
y_data = X[:, 1]

betaA5 = regfreqA(X, 5)
betaB5 = regfreqB(X, 5)
betaA15 = regfreqA(X, 15)
betaB15 = regfreqB(X, 15)

def f_freq(x, beta, k):
    y = beta[0, 0] * np.ones_like(x)

    for j in range(1, k):
        y += beta[j, 0] * np.cos(j * x)

    for j in range(1, k):
        y += beta[k - 1 + j, 0] * np.sin(j * x)

    return y

x = np.linspace(1, 6, 400)

yA5 = f_freq(x, betaA5, 5)
yB5 = f_freq(x, betaB5, 5)
yA15 = f_freq(x, betaA15, 15)
yB15 = f_freq(x, betaB15, 15)

plt.figure()
plt.scatter(x_data, y_data, s=5, label="Données", c='plum')
plt.plot(x, yA5, label="Régression de Fourier A (k=5)")
plt.plot(x, yB5, label="Régression de Fourier B (k=5)")
plt.plot(x, yA15, label="Régression de Fourier A (k=15)")
plt.plot(x, yB15, label="Régression de Fourier B (k=15)")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Figure 4: Régression de Fourier")
plt.legend()
plt.grid()
plt.show()


# ===== k) Temps de calcul =====

def temps_moyen(f, X, k):
    t0 = time()
    for _ in range(1000):
        f(X, k)
    return (time() - t0) / 1000

tA5 = temps_moyen(regfreqA, X, 5)
tB5 = temps_moyen(regfreqB, X, 5)
tA15 = temps_moyen(regfreqA, X, 15)
tB15 = temps_moyen(regfreqB, X, 15)

print("Temps moyen A k=5 :", tA5)
print("Temps moyen B k=5 :", tB5)
print("Temps moyen A k=15 :", tA15)
print("Temps moyen B k=15 :", tB15)
