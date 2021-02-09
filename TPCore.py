import numpy as np, numpy.random as random, numpy.linalg as la
import utils

# This is a helper file which runs the core TP updates used in various algorithms.
# Main functions below are TPAlgosampledRows and TPAlgo

# -----------------------------
# TP Plain Algo
# Description: Takes as input xk, A and b and outputs x_k+1
# Note 1: A and b have same number of rows
# Note 2: A has same number of columns as xk
# Note 3: This can also be thought of as a distance weighted update
# Update Step:
# x_k+1 = xk + 1/n A_i (A_i x_k -b_i)
#
# -----------------------------
def TPAlgoDW(A, b, xk):
    n = len(A)
    x_kPlusOne = np.array(xk, copy=True)
    for i in range(n):
        Ai = A[i, :]
        x_kPlusOne = x_kPlusOne - 1 / n * Ai * (Ai.dot(xk) - b[i])
    return x_kPlusOne


# -----------------------------
# TP Plain Algo
# Description: Takes as input xk, A and b and outputs x_k+1
# Note 1: A and b have same number of rows
# Note 2: A has same number of columns as xk
# Update Step:
# x_k+1 = xk + 1/n A_i (A_i x_k -b_i)/||A_i||^2
# -----------------------------

def TPAlgo(A, b, xk):
    n = len(A)
    x_kPlusOne = np.array(xk, copy=True)
    for i in range(n):
        Ai = A[i, :]
        x_kPlusOne = x_kPlusOne - 1 / n * Ai * (Ai.dot(xk) - b[i]) / utils.twoNormSq(Ai)
    return x_kPlusOne

# -----------------------------
# TP Plain Algo with Row Choice
# Description: Takes as input xk, A and b and outputs x_k+1
# Note 1: A and b have same number of rows
# Note 2: A has same number of columns as xk
# Update Step:
# x_k+1 = xk + 1/n A_i (A_i x_k -b_i)/||A_i||^2
# -----------------------------

def TPAlgosampledRows(A, b, xk,sampledRows):
    m = len(A)
    x_kPlusOne = np.array(xk, copy=True)
    for i in sampledRows:
        if i<m:
            Ai = A[i, :]
            x_kPlusOne = x_kPlusOne - 1 / m * Ai * (Ai.dot(xk) - b[i]) / utils.twoNormSq(Ai)

        else:
            print("the row input - %d - is larger than the number of rows %d in matrix A"%(i,m))
    return x_kPlusOne


# -----------------------------
# TP LeastSq
# Description: Takes as input xk, A and b and outputs x_k+1
# Note 1: A and b have same number of rows
# Note 2: A has same number of columns as xk
# Note 3: We modify Kaczmarz to have square
# Update Step:
# x_k+1 = xk + 1/n A_i (A_i x_k -b_i)/||A_i||^2
# -----------------------------

def TPAlgoNormDistance(A, b, xk, norm):
    m = len(A)
    x_kPlusOne = np.array(xk, copy=True)
    for i in range(m):
        Ai = A[i, :]
        x_kPlusOne = x_kPlusOne - (Ai * (Ai.dot(xk) - b[i]) * (utils.twoNorm(Ai.dot(xk) - b[i])**(norm-2))) / (m**(norm-1)*((utils.twoNorm(Ai))**norm))
    return x_kPlusOne
