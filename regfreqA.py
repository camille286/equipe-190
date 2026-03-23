import numpy as np
from scipy.linalg import lu
from numpy.linalg import solve, norm

def regfreqA(X, k, afficher=True):
    '''
    Args:
        X : matrice n x 2 contenant les points (xi, yi)
        k : nombre de frequences

    Returns:
        beta : vecteur colonne des coefficients
    '''

    x = X[:, 0]
    y = X[:, 1].reshape(-1, 1)
    n = len(x)

    # Construction de la matrice A de taille n x (2k-1)
    A = np.ones((n, 2*k - 1))

    # Colonnes cos(jx), j = 1, ..., k-1
    for j in range(1, k):
        A[:, j] = np.cos(j * x)

    # Colonnes sin(jx), j = 1, ..., k-1
    for j in range(1, k):
        A[:, k - 1 + j] = np.sin(j * x)

    # Approche A : equations normales
    ATA = A.T @ A
    ATy = A.T @ y

    P, L, U = lu(ATA)
    z = solve(L, P.T @ ATy)
    beta = solve(U, z)

    res = norm(A @ beta - y)
    cond = np.linalg.cond(A.T @ A)

    if afficher:
        print(f"Norme du residu ||F(beta)|| = ||A*beta - y|| = {res}")
        print(f"cond(A^T A) = {cond}")

    return beta
