#cython: language_level=3
#cython: linetrace=True
import numpy as np
import scipy.sparse as sp

from libc.stdint cimport uint8_t
from libcpp cimport bool

cdef extern from 'Accelerate/Accelerate.h':
    enum SparseKind_t:
        SparseOrdinary
        SparseTriangular
        SparseUnitTriangular
        SparseSymmetric

    enum SparseTriangle_t:
        SparseUpperTriangle
        SparseLowerTriangle

    ctypedef struct SparseAttributes_t:
        bool            transpose
        SparseTriangle_t triangle
        SparseKind_t         kind
        unsigned int    _reserved
        bool   _allocatedBySparse

    ctypedef struct SparseMatrixStructure:
        int rowCount;
        int columnCount;
        long *columnStarts;
        int *rowIndices;
        SparseAttributes_t attributes;
        uint8_t blockSize;

    ctypedef struct SparseMatrix_Double:
        SparseMatrixStructure structure
        double *data

    ctypedef struct SparseMatrix_Float:
        SparseMatrixStructure structure
        float *data

    ctypedef struct DenseVector_Double:
          int count
          double *data

    ctypedef struct DenseVector_Float:
        int count
        float *data

    ctypedef struct DenseMatrix_Double:
        int rowCount
        int columnCount
        int columnStride
        SparseAttributes_t attributes
        double *data

    ctypedef struct DenseMatrix_Float:
        int rowCount
        int columnCount
        int columnStride
        SparseAttributes_t attributes
        float *data

    enum SparseStatus_t:
        SparseStatusOK
        SparseFactorizationFailed
        SparseMatrixIsSingular
        SparseInternalError
        SparseParameterError
        SparseStatusReleased

    enum SparseFactorization_t:
        SparseFactorizationCholesky
        SparseFactorizationLDLT
        SparseFactorizationLDLTUnpivoted
        SparseFactorizationLDLTSBK
        SparseFactorizationLDLTTPP
        SparseFactorizationQR
        SparseFactorizationCholeskyAtA

    enum SparseControl_t:
      SparseDefaultControl

    enum SparseOrder_t:
        SparseOrderDefault
        SparseOrderUser
        SparseOrderAMD
        SparseOrderMetis
        SparseOrderCOLAMD

    enum SparseScaling_t:
        SparseScalingDefault
        SparseScalingUser
        SparseScalingEquilibriationInf

    ctypedef struct SparseSymbolicFactorOptions:
        SparseControl_t control
        SparseOrder_t orderMethod
        int * order
        int * ignoreRowsAndColumns
        void * (* malloc)(size_t size)
        void (* free)(void * pointer)
        void (* reportError)(const char *message)

    ctypedef struct SparseNumericFactorOptions:
        SparseControl_t control
        SparseScaling_t scalingMethod
        void * scaling
        double pivotTolerance
        double zeroTolerance

    ctypedef struct SparseOpaqueSymbolicFactorization:
        SparseStatus_t status
        int rowCount
        int columnCount
        SparseAttributes_t attributes
        uint8_t blockSize
        SparseFactorization_t type
        void * factorization
        size_t workspaceSize_Float
        size_t workspaceSize_Double
        size_t factorSize_Float
        size_t factorSize_Double

    ctypedef struct SparseOpaqueFactorization_Double:
        SparseStatus_t status
        SparseAttributes_t attributes
        SparseOpaqueSymbolicFactorization symbolicFactorization
        bool userFactorStorage
        void * numericFactorization
        size_t solveWorkspaceRequiredStatic
        size_t solveWorkspaceRequiredPerRHS

    ctypedef struct SparseOpaqueFactorization_Float:
        SparseStatus_t status
        SparseAttributes_t attributes
        SparseOpaqueSymbolicFactorization symbolicFactorization
        bool userFactorStorage
        void * numericFactorization
        size_t solveWorkspaceRequiredStatic
        size_t solveWorkspaceRequiredPerRHS

    SparseOpaqueFactorization_Double SparseFactor(SparseFactorization_t type, SparseMatrix_Double Matrix)
    # SparseOpaqueFactorization_Double SparseFactor(SparseOpaqueSymbolicFactorization, SparseMatrix_Double, SparseNumericFactorOptions)
    # SparseOpaqueFactorization_Float SparseFactor(SparseFactorization_t type, SparseMatrix_Float Matrix)

    # void SparseSolve(SparseOpaqueFactorization_Double Factored, DenseMatrix_Double XB)
    void SparseSolve(SparseOpaqueFactorization_Double Factored, DenseVector_Double b, DenseVector_Double x)
    # void SparseSolve(SparseOpaqueFactorization_Float Factored, DenseMatrix_Float XB)
    # void SparseSolve(SparseOpaqueFactorization_Float Factored, DenseMatrix_Float B, DenseMatrix_Float X)

    void SparseMultiplyAdd(SparseMatrix_Double, DenseVector_Double, DenseVector_Double);

FACTOR_TYPES = {
    "cholesky" : 0,
    "ldlt" : 1,
    "ldlt-unpivoted" : 2,
    "ldlt-sbk" : 3,
    "ldlt-tpp" : 4,
    "qr" : 5,
    "cholesky-ata" : 6,
}


cdef class AccelerateSolver:
    cdef:
        SparseMatrix_Double _A
        long[:] col_start
        int[:] row_inds
        double[:] A_data
        SparseOpaqueFactorization_Double _factor
        int _algorithm
        bool factored
        bool _complex

    def __init__(self, A, factor_type=None):
        if np.issubdtype(A.dtype, np.complexfloating):
            # We can replace A with another matrix that is
            # (A + j * B) * (x_r + j * x_i) = (b_r + j * b_i)
            # which breaks out system of equations for a complex matrix
            # into:
            # [A, -B] [x_r] = [b_r]
            # [B,  A] [x_i] = [b_i]
            # But we exchange the top and bottom rows to preserve any
            # symmetric aspect of the underlying matrix
            self._complex = True
            A_r = A.real
            A_i = A.imag

            A = sp.bmat([[A_i, A_r],[A_r, -A_i]], format='csc')
            if factor_type is None:
                factor_type = 'ldlt-sbk'
        else:
            self._complex = False
            if factor_type is None:
                factor_type = "cholesky"
        self._algorithm = FACTOR_TYPES[factor_type]
        if self._algorithm < 5:
            A = sp.tril(A, format='csc')
        else:
            A = A.tocsc()

        # if
        m, n = A.shape
        self._A.structure.rowCount = m
        self._A.structure.columnCount = n
        # column
        self.col_start = A.indptr.astype(np.int_)
        self.row_inds = A.indices.astype(np.intc)
        self._A.structure.columnStarts = &self.col_start[0]
        self._A.structure.rowIndices = &self.row_inds[0]

        cdef SparseAttributes_t attr
        attr.transpose = False
        if self._algorithm < 5:
            attr.kind = SparseKind_t.SparseSymmetric
            attr.triangle = SparseTriangle_t.SparseLowerTriangle
        self._A.structure.attributes = attr
        self._A.structure.blockSize = 1

        self.A_data = A.data.astype(np.double)
        self._A.data = &self.A_data[0]
        self.factored = False
        self.factor()

    def factor(self):
        if not self.factored:
            if self._algorithm == 0:
                self._factor = SparseFactor(SparseFactorizationCholesky, self._A)
            elif self._algorithm == 1:
                self._factor = SparseFactor(SparseFactorizationLDLT, self._A)
            elif self._algorithm == 2:
                self._factor = SparseFactor(SparseFactorizationLDLTUnpivoted, self._A)
            elif self._algorithm == 3:
                self._factor = SparseFactor(SparseFactorizationLDLTSBK, self._A)
            elif self._algorithm == 4:
                self._factor = SparseFactor(SparseFactorizationLDLTTPP, self._A)
            elif self._algorithm == 5:
                self._factor = SparseFactor(SparseFactorizationQR, self._A)
            elif self._algorithm == 6:
                self._factor = SparseFactor(SparseFactorizationCholeskyAtA, self._A)
        self.factored = True

    def solve(self, rhs, out=None, refinement_steps=1):
        if self._complex:
            # TODO ensure rhs is complex

            # A complex number is a pair of two floats
            # this lets us look at them all as floats ordered as x_r_1, x_i_1, x_r_2, x_i_2...
            rhs = rhs.view(np.float64)
            rhs = rhs.reshape((-1, 2))
            # rhs is (n, 2) first column is real, second is imaginary

            # For the system of equations we need to reverse last dimension
            # transpose it, and flatten it
            rhs = rhs[:, ::-1].T.reshape(-1)
            # this should be ordered as np.r_[rhs.imag, rhs.real]
        cdef double[:] b = np.require(rhs, dtype=np.double, requirements='C')
        cdef double[:] x = np.empty(len(rhs), np.double)
        cdef double[:] r
        cdef double[:] corr

        cdef int n = len(rhs)
        cdef int i, j

        cdef DenseVector_Double B, X, R, C
        B.count = n
        B.data = &b[0]

        X.count = n
        X.data = &x[0]

        SparseSolve(self._factor, B, X)

        if refinement_steps > 1:
            _r = np.empty(len(rhs), np.double)
            r = _r
            corr = np.empty(len(rhs), np.double)

            R.count = n
            R.data = &r[0]
            C.count = n
            C.data = &corr[0]

            for i in range(refinement_steps-1):
                # R = A*X - B

                for j in range(n):
                    r[j] = -b[j]

                SparseMultiplyAdd(self._A, X, R)
                SparseSolve(self._factor, R, C)

                for j in range(n):
                    x[j] -= corr[j]

        out = np.array(x)
        if self._complex:
            # this should do the equivalent as out = out[:n] + 1j * out[n:]
            # but without multiplying and adding and type casting.
            out = out.reshape([2, -1]).T.reshape(-1)
            out = out.view(np.complex128)
        return out
