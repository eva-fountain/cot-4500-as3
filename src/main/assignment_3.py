# Eva Fountain
# COT4500, Spring 2023
# Assignment 3

import numpy as np
import sympy as sp

# Number 1

def function(t: float, y: float):
    return t - (y**2)

def euler(t0, y, h, t):
    temp = 0

    while t0 < t:
        temp = y
        y = y + h * function(t0, y)
        t0 = t0 + h

    # Print
    print("%.5f" % y)

# Initial values
t0 = 0
y0 = 1
start_of_t = 0
end_of_t = 2
iterations = 10
h = (end_of_t - start_of_t) / iterations

# Approximation value
t = 1.80

euler(t0, y0, h, t)
print("\n")

# Number 2

def dydx(t, y):
    return t - (y**2)

def runge_kutta(t0, y0, t, h):
    n = (int)((t - t0)/h)
    y = y0
    for i in range(1, n + 1):
        k1 = h * dydx(t0, y)
        k2 = h * dydx(t0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * dydx(t0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * dydx(t0 + h, y + k3)

        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        t0 = t0 + h

    return y

t0 = 0
y = 1
t = 2
print("%.5f " % runge_kutta(t0, y, t, h))
print("\n")

# Number 3

def gauss(a,b):
    n = len(A)

    for i in range(n):
        # Finding pivot
        pivot = abs(A[i][i])
        pivot_row = i
        for j in range(i+1, n):
            if abs(A[i][j]) > pivot:
                pivot = abs(A[j][i])
                pivot_row = j

        # Swap pivot row and current row if needed
        if pivot_row != i:
            A[i], A[pivot_row] = A[pivot_row], A[i]
            b[i], b[pivot_row] = b[pivot_row], b[i]

        # Eliminate current variable
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]

    # Backwards substitution
    x = [0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]
    return x

# Initial
A = [[2, -1, 1], [1, 3, 1], [-1, 5, 4]]
b = [6, 0, -3]

x = gauss(A, b)
matrix = np.array(x, dtype=np.double)
print(matrix)
print("\n")

# Number 4

# Determinant
def determ(matrix):
    return np.linalg.det(matrix)

# Couldn't quite figure out LU factorization

matrix = np.array([[1, 1, 0, 3],
                  [2, 1, -1, 1],
                  [3, -1, -1, 2],
                  [-1, 2, 3, -1]])
print("%.5f" % np.array(determ(matrix)))

print("\n")
# Number 5

def check(a, b):
    for i in range(0, b):
        row_sum = 0
        for j in range (0, b):
            row_sum = row_sum + abs(a[i][j])

        # Remove diagonal element
        row_sum = row_sum - abs(a[i][j])

        # is diagonal element less than sum of non-diagonal element?
        if(abs(a[i][i]) < row_sum):
            return False
    return True


b = 5
a = [[9, 0, 5, 2, 1],
     [3, 9, 1, 2, 1],
     [0, 1, 7, 2, 3],
     [4, 2, 3, 12, 2],
     [3, 2, 4, 0, 8]]

if(check(a, b)):
    print("True")
else:
    print("False")

print("\n")

# Number 6

def is_pos_def(A):
    return np.all(np.linalg.eigvals(A+A.transpose()) > 0)


A = np.array([[2, 2, 1],
            [2, 3, 0],
            [1, 0, 2]])

print(is_pos_def(A))
