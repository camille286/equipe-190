import numpy as np
from numpy.linalg import solve, norm

def regfreqB(X, k):
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

    # Approche B : QR
    Q, R = np.linalg.qr(A)
    beta = solve(R, Q.T @ y)

    res = norm(A @ beta - y)
    print(f"Norme du residu ||F(beta)|| = ||A*beta - y|| = {res}")

    return beta