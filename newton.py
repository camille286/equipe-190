import numpy as np
from numpy.linalg import solve, norm
import scipy.linalg as la

def newton(beta_init, F, J, tol, nmax):
    '''
    Args:
        beta_init : point initial (vecteur colonne 3x1)
        F         : fonction vectorielle (vecteur colonne nx1) dont on cherche le zero. DEPEND DE BETA.
        J         : matrice jacobienne de F (matrice nx3). DEPEND DE BETA.
        tol       : tolerance pour determiner la convergence
        nmax      : nombre maximal d'iterations

    Returns:
        beta      : vecteur des coefficients de la courbe de regression
    '''

    beta = beta_init.copy()
    assert beta.shape == (3,1), "beta_init doit être un vecteur colonne!"
    n = 0
    res = norm(F(beta.flatten()))
    dbeta = np.inf

    while res > tol and norm(dbeta) > tol and n < nmax:
        Jb = J(beta.flatten())
        Fb = F(beta.flatten())
        assert Fb.shape == (1000,1), "F doit retourner un vecteur colonne!"

        # Resolution du systeme pour calculer la correction
        # Approche (A) : equations normales
        A = Jb
        b = -Fb

        ATA = A.T @ A
        ATb = A.T @ b

        dbeta = solve(ATA, ATb)

        # Appliquer la correction
        assert dbeta.shape == (3,1), "dbeta doit être un vecteur colonne!"
        beta += dbeta

        # Calcul du residu au nouveau point
        res = norm(F(beta.flatten()))
        print(f"Iteration {n} : ||dbeta|| = {norm(dbeta)}, ||F(beta)|| = {res}")
        n += 1

    return beta