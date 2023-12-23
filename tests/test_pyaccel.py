import numpy as np
import pytest
import scipy.sparse as sp
from pyaccel.accel_solver import AccelerateSolver

# from concurrent.futures import ThreadPoolExecutor
# import pytest
# import sys

np.random.seed(12345)
n = 40
L = sp.diags([-1, 1], [-1, 0], (n, n))
U = sp.diags([2, -1], [0, 1], (n, n))

e = np.ones(n)
e[0] = -1
D = sp.diags(e)

e = np.ones(n-2)
e[0] = -1
Da = sp.diags(e)  # diagonal matrix of 1 and -1
Db = np.array([[-1, 1],
               [1, 1]])
D2 = sp.bmat([[Da, None],[None, Db]])

U2 = sp.diags([2, -1], [0, 2], (n, n))

Lc = sp.diags([-(1+1j), (1+1j)], [-1, 0], (n, n))
Uc = sp.diags([(2+2j), -(1+1j)], [0, 1], (n, n))
U2c = sp.diags([(2+2j), -(1+1j)], [0, 2], (n, n))

def test_solver_chol():
    A = L @ L.T
    x = np.arange(n).astype(np.float64)
    b = A @ x

    solver = AccelerateSolver(A, factor_type='cholesky')
    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=1E3*eps)
def test_double_solver_ldlt():
    A = L @ D @ L.T
    x = np.arange(n).astype(np.float64)
    b = A @ x

    solver = AccelerateSolver(A, factor_type='ldlt')
    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=1E3*eps)

def test_double_solver_ldlt_tpp():
    A = L @ D @ L.T
    x = np.arange(n).astype(np.float64)
    b = A @ x

    solver = AccelerateSolver(A, factor_type='ldlt-tpp')
    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=1E3*eps)

def test_double_solver_ldlt_sbk():
    A = L @ D2 @ L.T
    x = np.arange(n).astype(np.float64)
    b = A @ x

    solver = AccelerateSolver(A, factor_type='ldlt-sbk')
    x2 = solver.solve(b, refinement_steps=2)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=1E3*eps)

def test_double_solver_qr():
    A = L @ D2 @ U2
    x = np.arange(n).astype(np.float64)
    b = A @ x

    solver = AccelerateSolver(A, factor_type='qr')
    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=1E3*eps)

@pytest.mark.xfail
def test_double_solver_AtA():
    A = D @ L
    A_full = A.T @ A
    x = np.arange(n).astype(np.float64)
    b = A_full @ x

    solver = AccelerateSolver(A, factor_type='cholesky-ata')
    x2 = solver.solve(b, refinement_steps=2)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=1E3*eps)

def test_complex_solver_ldlt():
    A = Lc @ D @ Lc.T
    x = np.arange(n).astype(np.complex128)
    b = A @ x

    solver = AccelerateSolver(A, factor_type='ldlt')
    x2 = solver.solve(b, refinement_steps=2)
    eps = np.finfo(np.complex128).eps
    np.testing.assert_allclose(x, x2, atol=1E3*eps)

def test_complex_solver_ldlt_tpp():
    A = Lc @ D @ Lc.T
    x = np.arange(n).astype(np.float64)
    b = A @ x

    solver = AccelerateSolver(A, factor_type='ldlt-tpp')
    x2 = solver.solve(b, refinement_steps=2)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=1E3*eps)

def test_complex_solver_ldlt_sbk():
    A = Lc @ D2 @ Lc.T
    x = np.arange(n).astype(np.float64)
    b = A @ x

    solver = AccelerateSolver(A, factor_type='ldlt-sbk')
    x2 = solver.solve(b, refinement_steps=2)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=1E3*eps)

def test_complex_solver_qr():
    A = Lc @ D2 @ U2c
    x = np.arange(n).astype(np.float64)
    b = A @ x

    solver = AccelerateSolver(A, factor_type='qr')
    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=1E3*eps)