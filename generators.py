import numpy as np

# =====================================================================
# 1. PARAMETRIC ORTHOGONAL DESIGN (OD) GENERATORS
# =====================================================================

def X2(a1, a2):
    return np.array([[a1, a2], [-a2, a1]])

def X4(a1, a2, a3, a4):
    return np.array([[ a1,  a2,  a3,  a4],
                     [-a2,  a1, -a4,  a3],
                     [-a3,  a4,  a1, -a2],
                     [-a4, -a3,  a2,  a1]])

def X6a(a, b):
    return np.array([[ a,  b, -a,  a,  0,  a],
                     [-a,  a,  b,  a,  a,  0],
                     [ b, -a,  a,  0,  a,  a],
                     [-a, -a,  0,  a, -a,  b],
                     [ 0, -a, -a,  b,  a, -a],
                     [-a,  0, -a, -a,  b,  a]])

def X6b(a, b):
    return np.array([[ a,  0,  b, -a,  0,  b],
                     [ b,  a,  0,  b, -a,  0],
                     [ 0,  b,  a,  0,  b, -a],
                     [ a, -b,  0,  a,  b,  0],
                     [ 0,  a, -b,  0,  a,  b],
                     [-b,  0,  a,  b,  0,  a]])

def X6c(a, b):
    return np.array([[ a,  0,  0,  b,  0,  0],
                     [ 0,  a,  0,  0,  b,  0],
                     [ 0,  0,  a,  0,  0,  b],
                     [-b,  0,  0,  a,  0,  0],
                     [ 0, -b,  0,  0,  a,  0],
                     [ 0,  0, -b,  0,  0,  a]])

def X8(a1, a2, a3, a4, a5, a6, a7, a8):
    return np.array([[ a1,  a2,  a4,  a3,  a6,  a5,  a8,  a7],
                     [-a2,  a1,  a3, -a4,  a5, -a6,  a7, -a8],
                     [-a4, -a3,  a1,  a2, -a8,  a7,  a6, -a5],
                     [-a3,  a4, -a2,  a1,  a7,  a8, -a5, -a6],
                     [-a6, -a5,  a8, -a7,  a1,  a2, -a4,  a3],
                     [-a5,  a6, -a7, -a8, -a2,  a1,  a3,  a4],
                     [-a8, -a7, -a6,  a5,  a4, -a3,  a1,  a2],
                     [-a7,  a8,  a5,  a6, -a3, -a4, -a2,  a1]])

def X12a(a1, a2, a3, a4):
    return np.array([
        [ a4,  a2, -a2, -a3,  a3,  a1,  a3,  a3,  a2,  a2,  a2, -a3],
        [-a2,  a4,  a2,  a3,  a1, -a3,  a3,  a2,  a3,  a2, -a3,  a2],
        [ a2, -a2,  a4,  a1, -a3,  a3,  a2,  a3,  a3, -a3,  a2,  a2],
        [ a3, -a3, -a1,  a4,  a2, -a2, -a2, -a2,  a3,  a3,  a3,  a2],
        [-a3, -a1,  a3, -a2,  a4,  a2, -a2,  a3, -a2,  a3,  a2,  a3],
        [-a1,  a3, -a3,  a2, -a2,  a4,  a3, -a2, -a2,  a2,  a3,  a3],
        [-a3, -a3, -a2,  a2,  a2, -a3,  a4,  a2, -a2, -a3,  a3, -a1],
        [-a3, -a2, -a3,  a2, -a3,  a2, -a2,  a4,  a2,  a3, -a1, -a3],
        [-a2, -a3, -a3, -a3,  a2,  a2,  a2, -a2,  a4, -a1, -a3,  a3],
        [-a2, -a2,  a3, -a3, -a3, -a2,  a3, -a3,  a1,  a4,  a2, -a2],
        [-a2,  a3, -a2, -a3, -a2, -a3, -a3,  a1,  a3, -a2,  a4,  a2],
        [ a3, -a2, -a2, -a2, -a3, -a3,  a1,  a3, -a3,  a2, -a2,  a4]
    ])

def X12b(a1, a2, a3, a4):
    return np.array([
        [ a4,  a3, -a3, -a3,  a3,  a1,  a3,  a3,  a2,  a3,  a3, -a2],
        [-a3,  a4,  a3,  a3,  a1, -a3,  a3,  a2,  a3,  a3, -a2,  a3],
        [ a3, -a3,  a4,  a1, -a3,  a3,  a2,  a3,  a3, -a2,  a3,  a3],
        [ a3, -a3, -a1,  a4,  a3, -a3, -a3, -a3,  a2,  a3,  a3,  a2],
        [-a3, -a1,  a3, -a3,  a4,  a3, -a3,  a2, -a3,  a3,  a2,  a3],
        [-a1,  a3, -a3,  a3, -a3,  a4,  a2, -a3, -a3,  a2,  a3,  a3],
        [-a3, -a3, -a2,  a3,  a3, -a2,  a4,  a3, -a3, -a3,  a3, -a1],
        [-a3, -a2, -a3,  a3, -a2,  a3, -a3,  a4,  a3,  a3, -a1, -a3],
        [-a2, -a3, -a3, -a2,  a3,  a3,  a3, -a3,  a4, -a1, -a3,  a3],
        [-a3, -a3,  a2, -a3, -a3, -a2,  a3, -a3,  a1,  a4,  a3, -a3],
        [-a3,  a2, -a3, -a3, -a2, -a3, -a3,  a1,  a3, -a3,  a4,  a3],
        [ a2, -a3, -a3, -a2, -a3, -a3,  a1,  a3, -a3,  a3, -a3,  a4]
    ])

def X12c(a1, a2, a3, a4):
    return np.array([
        [ a4,  a1, -a1, -a1,  a1,  a2, -a1,  a1,  a3,  a1,  a1,  a1],
        [-a1,  a4,  a1,  a1,  a2, -a1,  a1,  a3, -a1,  a1,  a1,  a1],
        [ a1, -a1,  a4,  a2, -a1,  a1,  a3, -a1,  a1,  a1,  a1,  a1],
        [ a1, -a1, -a2,  a4,  a1, -a1, -a1, -a1, -a1,  a1, -a1,  a3],
        [-a1, -a2,  a1, -a1,  a4,  a1, -a1, -a1, -a1, -a1,  a3,  a1],
        [-a2,  a1, -a1,  a1, -a1,  a4, -a1, -a1, -a1,  a3,  a1, -a1],
        [ a1, -a1, -a3,  a1,  a1,  a1,  a4,  a1, -a1, -a1,  a1, -a2],
        [-a1, -a3,  a1,  a1,  a1,  a1, -a1,  a4,  a1,  a1, -a2, -a1],
        [-a3,  a1, -a1,  a1,  a1,  a1,  a1, -a1,  a4, -a2, -a1,  a1],
        [-a1, -a1, -a1, -a1,  a1, -a3,  a1, -a1,  a2,  a4,  a1, -a1],
        [-a1, -a1, -a1,  a1, -a3, -a1, -a1,  a2,  a1, -a1,  a4,  a1],
        [-a1, -a1, -a1, -a3, -a1,  a1,  a2,  a1, -a1,  a1, -a1,  a4]
    ])

def X16(a, b, c, d, e, f, g, h):
    return np.array([
        [ a,  b,  b,  b, -c,  d,  d,  d, -e,  f,  f,  f, -g,  h,  h,  h],
        [-b,  a, -b,  b,  d,  d, -d,  c,  f,  f, -f,  e,  h,  h, -h,  g],
        [-b,  b,  a, -b,  d, -d,  c,  d,  f, -f,  e,  f,  h, -h,  g,  h],
        [-b, -b,  b,  a,  d,  c,  d, -d,  f,  e,  f, -f,  h,  g,  h, -h],
        [ c, -d, -d, -d,  a,  b,  b,  b, -g, -h, -h, -h,  e,  f,  f,  f],
        [-d, -d,  d, -c, -b,  a, -b,  b, -h, -h,  h,  g,  f,  f, -f, -e],
        [-d,  d, -c, -d, -b,  b,  a, -b, -h,  h,  g, -h,  f, -f, -e,  f],
        [-d, -c, -d,  d, -b, -b,  b,  a, -h,  g, -h,  h,  f, -e,  f, -f],
        [ e, -f, -f, -f,  g,  h,  h,  h,  a,  b,  b,  b, -c, -d, -d, -d],
        [-f, -f,  f, -e,  h,  h, -h, -g, -b,  a, -b,  b, -d, -d,  d,  c],
        [-f,  f, -e, -f,  h, -h, -g,  h, -b,  b,  a, -b, -d,  d,  c, -d],
        [-f, -e, -f,  f,  h, -g,  h, -h, -b, -b,  b,  a, -d,  c, -d,  d],
        [ g, -h, -h, -h, -e, -f, -f, -f,  c,  d,  d,  d,  a,  b,  b,  b],
        [-h, -h,  h, -g, -f, -f,  f,  e,  d,  d, -d, -c, -b,  a, -b,  b],
        [-h,  h, -g, -h, -f,  f,  e, -f,  d, -d, -c,  d, -b,  b,  a, -b],
        [-h, -g, -h,  h, -f,  e, -f,  f,  d, -c,  d, -d, -b, -b,  b,  a]
    ])

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

# =====================================================================
# 2. DYNAMIC CONFERENCE MATRICES SELECTOR
# =====================================================================

def get_conference_matrix(order):
    """
    Returns the specific core conference matrix (C_n) matched to 
    the active dimensional columns requested.
    """
    if order == 6:
        return np.array([
            [ 0,  1, -1, -1, -1, -1],
            [ 1,  0, -1,  1,  1, -1],
            [-1, -1,  0,  1, -1, -1],
            [-1,  1,  1,  0,  1, -1],
            [ 1, -1,  1, -1,  0, -1],
            [ 1,  1,  1,  1, -1,  0]
        ])
    elif order == 8:
        return np.array([
            [ 0,  1,  1,  1,  1,  1,  1,  1],
            [-1,  0, -1, -1, -1,  1,  1,  1],
            [-1,  1,  0,  1, -1, -1, -1,  1],
            [-1,  1, -1,  0,  1,  1, -1, -1],
            [-1,  1,  1, -1,  0, -1,  1, -1],
            [-1, -1,  1, -1,  1,  0, -1,  1],
            [-1, -1,  1,  1, -1,  1,  0, -1],
            [-1, -1, -1,  1,  1, -1,  1,  0]
        ])
    elif order == 10:
        return np.array([
            [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [ 1,  0, -1, -1, -1, -1,  1,  1,  1,  1],
            [ 1, -1,  0, -1,  1,  1, -1, -1,  1,  1],
            [ 1, -1, -1,  0,  1,  1,  1,  1, -1, -1],
            [ 1, -1,  1,  1,  0, -1, -1,  1, -1,  1],
            [ 1, -1,  1,  1, -1,  0,  1, -1,  1, -1],
            [ 1,  1, -1,  1, -1,  1,  0, -1, -1,  1],
            [ 1,  1, -1,  1,  1, -1, -1,  0,  1, -1],
            [ 1,  1,  1, -1, -1,  1, -1,  1,  0, -1],
            [ 1,  1,  1, -1,  1, -1,  1, -1, -1,  0]
        ])
    elif order == 12:
        return np.array([
            [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [ 1,  0, -1, -1, -1, -1,  1, -1,  1,  1,  1,  1],
            [ 1,  1,  0,  1,  1, -1,  1, -1, -1,  1, -1, -1],
            [ 1,  1, -1,  0,  1,  1, -1, -1, -1, -1,  1,  1],
            [ 1,  1, -1, -1,  0,  1, -1,  1,  1,  1, -1, -1],
            [ 1,  1,  1, -1, -1,  0,  1,  1, -1, -1,  1, -1],
            [ 1, -1, -1,  1,  1, -1,  0,  1,  1, -1,  1, -1],
            [ 1,  1,  1,  1, -1, -1, -1,  0,  1, -1, -1,  1],
            [ 1, -1,  1,  1, -1,  1, -1, -1,  0,  1,  1, -1],
            [ 1, -1, -1,  1, -1,  1,  1,  1, -1,  0, -1,  1],
            [ 1, -1,  1, -1,  1, -1, -1,  1, -1,  1,  0,  1],
            [ 1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  0]
        ])
    else:
        raise ValueError(f"Conference Matrix order {order} is not supported.")