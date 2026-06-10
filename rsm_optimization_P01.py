import os
import itertools
import numpy as np
from scipy.linalg import hadamard

# 1. PARAMETRIC ORTHOGONAL DESIGN (OD) GENERATORS
def X2(a1, a2):
    return np.array([[a1, a2], [-a2, a1]])

def X4(a1, a2, a3, a4):
    return np.array([[ a1,  a2,  a3,  a4],
                     [-a2,  a1, -a4,  a3],
                     [-a3,  a4,  a1, -a2],
                     [-a4, -a3,  a2,  a1]])

def X8(a1, a2, a3, a4, a5, a6, a7, a8):
    return np.array([[ a1,  a2,  a4,  a3,  a6,  a5,  a8,  a7],
                     [-a2,  a1,  a3, -a4,  a5, -a6,  a7, -a8],
                     [-a4, -a3,  a1,  a2, -a8,  a7,  a6, -a5],
                     [-a3,  a4, -a2,  a1,  a7,  a8, -a5, -a6],
                     [-a6, -a5,  a8, -a7,  a1,  a2, -a4,  a3],
                     [-a5,  a6, -a7, -a8, -2,  a1,  a3,  a4],
                     [-a8, -a7, -a6,  a5,  a4, -a3,  a1,  a2],
                     [-a7,  a8,  a5,  a6, -a3, -a4, -a2,  a1]])

def X20(a, b, c, d):
    return np.array([
        [ a,  c,  d, -d, -c, -d,  c, -c,  d,  b,  c,  c,  c,  c, -c,  d,  d,  d,  d, -d],
        [-c,  a,  c,  d, -d,  c, -c,  d,  b, -d,  c,  c,  c, -c,  c,  d,  d,  d, -d,  d],
        [-d, -c,  a,  c,  d, -c,  d,  b, -d,  c,  c,  c, -c,  c,  c,  d,  d, -d,  d,  d],
        [ d, -d, -c,  a,  c,  d,  b, -d,  c, -c,  c, -c,  c,  c,  c,  d, -d,  d,  d,  d],
        [ c,  d, -d, -c,  a,  b, -d,  c, -c,  d, -c,  c,  c,  c,  c, -d,  d,  d,  d,  d],
        [ d, -c,  c, -d, -b,  a,  c,  d, -d, -c, -d, -d, -d, -d,  d,  c,  c,  c,  c, -c],
        [-c,  c, -d, -b,  d, -c,  a,  c,  d, -d, -d, -d, -d,  d, -d,  c,  c,  c, -c,  c],
        [ c, -d, -b,  d, -c, -d, -c,  a,  c,  d, -d, -d,  d, -d, -d,  c,  c, -c,  c,  c],
        [-d, -b,  d, -c,  c,  d, -d, -c,  a,  c, -d,  d, -d, -d, -d,  c, -c,  c,  c,  c],
        [-b,  d, -c,  c, -d,  c,  d, -d, -c,  a,  d, -d, -d, -d, -d, -c,  c,  c,  c,  c],
        [-c, -c, -c, -c,  c,  d,  d,  d,  d, -d,  a,  c,  d, -d, -c, -d,  c, -c,  d, -b],
        [-c, -c, -c,  c, -c,  d,  d,  d, -d,  d, -c,  a,  c,  d, -d,  c, -c,  d, -b, -d],
        [-c, -c,  c, -c, -c,  d,  d, -d,  d,  d, -d, -c,  a,  c,  d, -c,  d, -b, -d,  c],
        [-c,  c, -c, -c, -c,  d, -d,  d,  d,  d,  d, -d, -c,  a,  c,  d, -b, -d,  c, -c],
        [ c, -c, -c, -c, -c, -d,  d,  d,  d,  d,  c,  d, -d, -c,  a, -b, -d,  c, -c,  d],
        [-d, -d, -d, -d,  d, -c, -c, -c, -c,  c,  d, -c,  c, -d,  b,  a,  c,  d, -d, -c],
        [-d, -d, -d,  d, -d, -c, -c, -c,  c, -c, -c,  c, -d,  b,  d, -c,  a,  c,  d, -d],
        [-d, -d,  d, -d, -d, -c, -c,  c, -c, -c,  c, -d,  b,  d, -c, -d, -c,  a,  c,  d],
        [-d,  d, -d, -d, -d, -c,  c, -c, -c, -c, -d,  b,  d, -c,  c,  d, -d, -c,  a,  c],
        [ d, -d, -d, -d, -d,  c, -c, -c, -c, -c,  b,  d, -c,  c, -d,  c,  d, -d, -c,  a]
    ])

# 2. HELPER FUNCTIONS
def MakeModelMatrix(X):
    n, m = X.shape
    D = np.hstack((X, X * X))
    interactions = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            interactions.append(X[:, i] * X[:, j])
    if interactions:
        D = np.hstack((D, np.column_stack(interactions)))
    D = np.hstack((np.ones((n, 1)), D))
    return D

def WriteDesignToFile(filename, D, Qstar, D_value):
    try:
        with open(filename, 'a') as f:
            f.write(f"Qstar: {Qstar:.6f} | D-value: {D_value:.10f}\n")
            np.savetxt(f, D, fmt='%+.2f')
            f.write("="*40 + "\n")
        return 1
    except:
        return 0

def MockRotatability(D):
    return 0.85

# 3. STRATEGY CONSTRUCTOR
def Construction3g2(Xf, XA, nc, s, fname, Qbest, Dbest, dynamic_type, critical_D):
    n_runs, m_factors = XA.shape
    C = np.zeros((nc, s))
    if s == 10 and m_factors >= 10:
        for idx in itertools.combinations(range(m_factors), 10):
            T = XA[:, list(idx)]
            D = np.vstack((Xf, T, C, -T))
            Xm = MakeModelMatrix(D)
            nn, p = Xm.shape
            try:
                det_val = np.abs(np.linalg.det(Xm.T @ Xm))
            except np.linalg.LinAlgError:
                det_val = 0.0
            D_value2 = ((10**3) * (det_val**(1/p))) / nn
            D_value1 = (10**9) * det_val / (nn**p)
            D_value = D_value1 if dynamic_type == 1 else D_value2
            Qstar = MockRotatability(D)
            if ((Qstar > Qbest) or (D_value > Dbest)) and (D_value > critical_D):
                if Qstar > Qbest: Qbest = Qstar
                if D_value > Dbest:
                    Dbest = D_value
                    WriteDesignToFile(fname, D, Qstar, D_value)
    return [Qbest, Dbest]

# 4. MAIN PIPELINE LOOP
def main():
    n, m, s, nc = 96, 10, 10, 0
    Qbest, Dbest1, Dbest2 = -1.0, -1.0, -1.0
    critical_D = 10e-300
    
    fnameD1 = f"out1_{n}_{m}_{s}_D3ge_Evang.txt"
    fnameD2 = f"out1_{n}_{m}_{s}_D3ge_Nguyen.txt"
    open(fnameD1, 'w').close()
    open(fnameD2, 'w').close()
    
    Xf1 = hadamard(32)
    Xf2 = Xf1[:, 1:m+1]
    
    C10 = np.array([
        [0,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [1,  0, -1, -1, -1, -1,  1,  1,  1,  1],
        [1, -1,  0, -1,  1,  1, -1, -1,  1,  1],
        [1, -1, -1,  0,  1,  1,  1,  1, -1, -1],
        [1, -1,  1,  1,  0, -1, -1,  1, -1,  1],
        [1, -1,  1,  1, -1,  0,  1, -1,  1, -1],
        [1,  1, -1,  1, -1,  1,  0, -1, -1,  1],
        [1,  1, -1,  1,  1, -1, -1,  0,  1, -1],
        [1,  1,  1, -1, -1,  1, -1,  1,  0, -1],
        [1,  1,  1, -1,  1, -1,  1, -1, -1,  0]
    ])
    f10 = np.vstack((C10, -1 * C10))
    Xf = np.vstack((Xf2, f10))
    
    binary_space = [0, 1]
    print("Running optimization matrices loops...")
    
    for p_set in itertools.product(binary_space, repeat=11):
        j1, j2, j3, j4, j5, j6, j7, j8, jb, j12, ja = p_set
        XT20 = X20(j1, j2, j3, j4)
        XT22 = np.block([
            [XT20, np.zeros((20, 2))],
            [np.zeros((2, 20)), X2(jb, j12)]
        ])
        XA = XT22[:, 0:m]
        
        res1 = Construction3g2(Xf, XA, nc, s, fnameD1, Qbest, Dbest1, 1, critical_D)
        Qbest, Dbest1 = res1[0], res1[1]
        res2 = Construction3g2(Xf, XA, nc, s, fnameD2, Qbest, Dbest2, 2, critical_D)
        Qbest, Dbest2 = res2[0], res2[1]

    print("Calculations complete inside browser.")

if __name__ == "__main__":
    main()