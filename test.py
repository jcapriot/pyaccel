import scipy.sparse as sp
import numpy as np
from pyaccel.accel_solver import AccelerateSolver
import discretize
import time
try:
    from pydiso.mkl_solver import MKLPardisoSolver
except ImportError:
    MKLPardisoSolver = None

n = 100

mesh = discretize.TensorMesh((n, n))
D = mesh.face_divergence
Mf = mesh.get_face_inner_product()

A = D @ Mf @ D.T

x = np.random.rand(mesh.n_cells)

b = A @ x

t1 = time.time()
Ainv = AccelerateSolver(A)
Ainv.factor()
t2 = time.time()
x2 = Ainv.solve(b)
t3 = time.time()
print("n:", n, 'nnz', A.nnz)
print("solve worked:", np.allclose(x, x2))
print("Accelerate factor:", t2-t1, "solve:", t3-t2)

if MKLPardisoSolver is not None:
    t1 = time.time()
    Ainv2 = MKLPardisoSolver(A, matrix_type='real_symmetric_positive_definite')

    t2 = time.time()
    x3 = Ainv2.solve(b)
    t3 = time.time()
    print("solve worked:", np.allclose(x, x3))
    print("MKL Pardiso factor:", t2-t1, "solve:", t3-t2)

C = mesh.edge_curl
Me = mesh.get_edge_inner_product()
Mfz = sp.diags(mesh.cell_volumes)
print(C.shape, Mfz.shape, Me.shape)
A_c = C.T @ Mfz @ C + 1j * Me

x_c = np.random.rand(A_c.shape[1]) + 1j * np.random.rand(A_c.shape[1])

b_c = A_c @ x_c

t1 = time.time()
Ainv3 = AccelerateSolver(A_c)
Ainv3.factor()
t2 = time.time()

t2 = time.time()
x_c2 = Ainv3.solve(b_c, refinement_steps=2)
t3 = time.time()

print('Solver worked:', np.allclose(x_c, x_c2))
print(f'Apple time, factor: {t2-t1}, solve: {t3-t2}')


if MKLPardisoSolver is not None:
    t1 = time.time()
    Ainv4 = MKLPardisoSolver(A_c, matrix_type='complex_symmetric')
    t2 = time.time()
    x_c3 = Ainv4.solve(b_c)
    t3 = time.time()
    print('Solver worked:', np.allclose(x_c, x_c3))
    print(f'MKL time, factor: {t2-t1}, solve: {t3-t2}')
