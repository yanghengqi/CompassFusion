"""
LAMBDA (Least-squares AMBiguity Decorrelation Adjustment) algorithm for
integer ambiguity resolution.

Reference:
[1] P.J.G.Teunissen, The least-square ambiguity decorrelation adjustment:
    a method for fast GPS ambiguity estimation, J.Geodesy, Vol.70, 65-82, 1995
[2] X.-W.Chang, X.Yang, T.Zhou, MLAMBDA: A modified LAMBDA method for
    integer least-squares estimation, J.Geodesy, Vol.79, 552-565, 2005

Ported from COMPASS C library (lambda.c)
"""

import numpy as np
from typing import Tuple, Optional

LOOPMAX = 10000  # Maximum count of search loop


def ld_factorization(Q: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    LD factorization: Q = L' * diag(D) * L
    
    Args:
        Q: Covariance matrix (n x n)
        
    Returns:
        L: Lower triangular matrix with unit diagonal (n x n)
        D: Diagonal elements (n,)
        Returns (None, None) if factorization fails
    """
    n = Q.shape[0]
    A = Q.copy()
    L = np.zeros((n, n))
    D = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        D[i] = A[i, i]
        if D[i] <= 0.0:
            return None, None
            
        a = np.sqrt(D[i])
        for j in range(i+1):
            L[i, j] = A[i, j] / a
            
        for j in range(i):
            for k in range(j+1):
                A[j, k] -= L[i, k] * L[i, j]
                
        for j in range(i+1):
            L[i, j] /= L[i, i]
            
    return L, D


def _gauss_transform(L: np.ndarray, Z: np.ndarray, i: int, j: int):
    """
    Integer Gauss transformation
    
    Args:
        L: Lower triangular matrix (modified in-place)
        Z: Transformation matrix (modified in-place)
        i, j: Indices for transformation
    """
    n = L.shape[0]
    mu = int(np.round(L[i, j]))
    
    if mu != 0:
        for k in range(i, n):
            L[k, j] -= mu * L[k, i]
        for k in range(n):
            Z[k, j] -= mu * Z[k, i]


def _permute(L: np.ndarray, D: np.ndarray, j: int, delta: float, Z: np.ndarray):
    """
    Permutation step in lambda reduction
    
    Args:
        L: Lower triangular matrix (modified in-place)
        D: Diagonal elements (modified in-place)
        j: Index for permutation
        delta: Computed delta value
        Z: Transformation matrix (modified in-place)
    """
    n = L.shape[0]
    
    eta = D[j] / delta
    lam = D[j+1] * L[j+1, j] / delta
    
    D[j] = eta * D[j+1]
    D[j+1] = delta
    
    for k in range(j):
        a0 = L[j, k]
        a1 = L[j+1, k]
        L[j, k] = -L[j+1, j] * a0 + a1
        L[j+1, k] = eta * a0 + lam * a1
        
    L[j+1, j] = lam
    
    # Swap columns in L and Z
    for k in range(j+2, n):
        L[k, j], L[k, j+1] = L[k, j+1], L[k, j]
    for k in range(n):
        Z[k, j], Z[k, j+1] = Z[k, j+1], Z[k, j]


def lambda_reduction(L: np.ndarray, D: np.ndarray, Z: np.ndarray):
    """
    LAMBDA reduction (z=Z'*a, Qz=Z'*Q*Z=L'*diag(D)*L)
    
    Args:
        L: Lower triangular matrix (modified in-place)
        D: Diagonal elements (modified in-place)
        Z: Transformation matrix (modified in-place)
    """
    n = L.shape[0]
    j = n - 2
    k = n - 2
    
    while j >= 0:
        if j <= k:
            for i in range(j+1, n):
                _gauss_transform(L, Z, i, j)
                
        delta = D[j] + L[j+1, j]**2 * D[j+1]
        
        # Compare considering numerical error
        if delta + 1e-6 < D[j+1]:
            _permute(L, D, j, delta, Z)
            k = j
            j = n - 2
        else:
            j -= 1


def mlambda_search(n: int, m: int, L: np.ndarray, D: np.ndarray, 
                   zs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modified LAMBDA search for integer least-squares
    
    Args:
        n: Number of ambiguities
        m: Number of solutions to find
        L: Transformed lower triangular matrix
        D: Transformed diagonal elements
        zs: Transformed float ambiguities
        
    Returns:
        zn: Fixed integer solutions (n x m)
        s: Sum of squared residuals for each solution (m,)
    """
    S = np.zeros((n, n))
    dist = np.zeros(n)
    zb = np.zeros(n)
    z = np.zeros(n)
    step = np.zeros(n)
    
    zn = np.zeros((n, m))
    s = np.full(m, 1e99)
    
    k = n - 1
    dist[k] = 0.0
    zb[k] = zs[k]
    z[k] = np.round(zb[k])
    y = zb[k] - z[k]
    step[k] = np.sign(y) if y != 0 else 1.0
    
    nn = 0
    imax = 0
    maxdist = 1e99
    
    for c in range(LOOPMAX):
        newdist = dist[k] + y * y / D[k]
        
        if newdist < maxdist:
            # Case 1: Move down
            if k != 0:
                k -= 1
                dist[k] = newdist
                
                for i in range(k+1):
                    S[k, i] = S[k+1, i] + (z[k+1] - zb[k+1]) * L[k+1, i]
                    
                zb[k] = zs[k] + S[k, k]
                z[k] = np.round(zb[k])
                y = zb[k] - z[k]
                step[k] = np.sign(y) if y != 0 else 1.0
                
            # Case 2: Store candidate and try next integer
            else:
                if nn < m:
                    if nn == 0 or newdist > s[imax]:
                        imax = nn
                    zn[:, nn] = z
                    s[nn] = newdist
                    nn += 1
                else:
                    if newdist < s[imax]:
                        zn[:, imax] = z
                        s[imax] = newdist
                        imax = np.argmax(s)
                    maxdist = s[imax]
                    
                z[0] += step[0]
                y = zb[0] - z[0]
                step[0] = -step[0] - np.sign(step[0])
                
        # Case 3: Exit or move up
        else:
            if k == n - 1:
                break
            else:
                k += 1
                z[k] += step[k]
                y = zb[k] - z[k]
                step[k] = -step[k] - np.sign(step[k])
                
    if c >= LOOPMAX - 1:
        raise RuntimeError("LAMBDA search loop count overflow")
        
    # Sort solutions by residuals (ascending)
    idx = np.argsort(s)
    zn = zn[:, idx]
    s = s[idx]
    
    return zn, s


def lambda_algorithm(a: np.ndarray, Q: np.ndarray, m: int = 2) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    LAMBDA integer least-squares estimation
    
    Args:
        a: Float ambiguity vector (n,)
        Q: Covariance matrix of float ambiguities (n x n)
        m: Number of fixed solutions to find (default: 2)
        
    Returns:
        F: Fixed integer solutions (n x m)
        s: Sum of squared residuals for each solution (m,)
        success: True if successful
    """
    n = len(a)
    
    if n == 0 or m <= 0:
        return np.array([]), np.array([]), False
        
    # LD factorization
    L, D = ld_factorization(Q)
    if L is None or D is None:
        return np.array([]), np.array([]), False
        
    # Initialize transformation matrix
    Z = np.eye(n)
    
    # LAMBDA reduction
    lambda_reduction(L, D, Z)
    
    # Transform float ambiguities
    z = Z.T @ a
    
    # MLAMBDA search
    try:
        E, s = mlambda_search(n, m, L, D, z)
    except RuntimeError:
        return np.array([]), np.array([]), False
        
    # z = Z.T @ a, so transform integer candidates with inv(Z.T).
    try:
        F = np.linalg.solve(Z.T, E)
    except np.linalg.LinAlgError:
        return np.array([]), np.array([]), False
    F = np.rint(F)
    
    return F, s, True


def lambda_ratio_test(s: np.ndarray, threshold: float = 3.0) -> bool:
    """
    Ratio test for ambiguity validation
    
    Args:
        s: Sum of squared residuals for candidate solutions
        threshold: Ratio threshold (default: 3.0)
        
    Returns:
        True if the best solution passes the ratio test
    """
    if len(s) < 2:
        return False
        
    if s[0] == 0.0:
        return True
        
    ratio = s[1] / s[0]
    return ratio >= threshold


def partial_lambda(a: np.ndarray, Q: np.ndarray, fixed_mask: np.ndarray, 
                   m: int = 2) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Partial LAMBDA: fix only a subset of ambiguities
    
    Args:
        a: Float ambiguity vector (n,)
        Q: Covariance matrix (n x n)
        fixed_mask: Boolean mask indicating which ambiguities to fix
        m: Number of solutions
        
    Returns:
        F: Fixed solutions with NaN for non-fixed ambiguities (n x m)
        s: Residuals
        success: Success flag
    """
    n = len(a)
    fix_idx = np.where(fixed_mask)[0]
    nf = len(fix_idx)
    
    if nf == 0:
        return np.full((n, m), np.nan), np.array([]), False
        
    # Extract subset to fix
    a_sub = a[fix_idx]
    Q_sub = Q[np.ix_(fix_idx, fix_idx)]
    
    # Run LAMBDA on subset
    F_sub, s, success = lambda_algorithm(a_sub, Q_sub, m)
    
    if not success:
        return np.full((n, m), np.nan), s, False
        
    # Reconstruct full solution
    F = np.full((n, m), np.nan)
    F[fix_idx, :] = F_sub
    
    return F, s, True
