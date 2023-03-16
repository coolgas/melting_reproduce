import numpy as np
import itertools

from scipy.special import zeta
from numba import jit

class RydbergIsingChainExact:
    def __init__(self, Omega, Delta, L):
        """Here we consider a finite chain with OBC.
        Math:
            H = \frac{\Omega}{2}\sum_{i}\sigma_{x}^{(i)}-\frac{\Delta}{2}\sum_{i}\sigma_{z}^{(i)}+\frac{1}{4}\sum_{i<j}(\sigma_{z}^{(i)}+\sigma_{z}^{(j)}+\sigma_{z}^{(i)}\sigma_{z}^{(j)})
        Inputs:
            Omega: Rabi frequency.
            Delta: Detuning.
            L: Chain length.
        """
        self.Omega = Omega
        self.Delta = Delta
        self.L = L

    @property
    def basis(self):
        combinations = np.array(list(itertools.product([1, 0], repeat=self.L)))
        return combinations
    
    @property
    def transverse_term(self):
        basis = self.basis
        L = self.L
        N = np.shape(basis)[0]
        H_trans = np.zeros((N, N), dtype=float)
        for i in range(L):
            basis_new = np.copy(basis)
            basis_new[:, i] = np.mod(np.add(basis_new[:, i], 1), 2)
            for j in range(N):
                basis_row_neg = basis_new[j, :]
                pos = np.where((basis == basis_row_neg).all(axis=1))[0]
                H_trans[pos, j] += 1.
        return H_trans
    
    @property
    def longitudinal_term1(self):
        basis = self.basis
        L = self.L
        N = np.shape(basis)[0]
        H_lgt = np.zeros((N, N), dtype=float)
        for i in range(L):
            for j in range(N):
                if basis[j][i] == 0.:
                    H_lgt[j][j] += -1.
                else:
                    H_lgt[j][j] += 1.
        return H_lgt
    
    @property
    def longitudinal_term2(self):
        _f = lambda x: 1/x**6 # the long-range interaction
        basis = self.basis
        L = self.L
        N = np.shape(basis)[0]
        H_lgt = np.zeros((N, N), dtype=float)
        for i in range(L-1):
            for j in range(i+1, L):
                for k in range(N):
                    val1 = basis[k][i]
                    val2 = basis[k][j]
                    val = val1+val2
                    if val == 2:
                        H_lgt[k][k] += 2.*_f(i-j)
                    elif val == 0:
                        H_lgt[k][k] += -2.*_f(i-j)
                    else:
                        H_lgt[k][k] += 0.
        return H_lgt

    @property
    def interaction_term(self):
        _f = lambda x: 1/x**6 # the long-range interaction
        basis = self.basis
        L = self.L
        N = np.shape(basis)[0]
        H_int = np.zeros((N, N))
        for i in range(L-1):
            for j in range(i+1, L):
                for k in range(N):
                    val1 = basis[k][i]
                    val2 = basis[k][j]
                    if val1 == val2:
                        H_int[k][k] += _f(i-j)*1.
                    else:
                        H_int[k][k] += _f(i-j)*-1.
        return H_int
    
    @property
    def H(self):
        Omega = self.Omega
        Delta = self.Delta
        H_trans = self.transverse_term
        H_lgt1 = self.longitudinal_term1
        H_lgt2 = self.longitudinal_term2
        H_int = self.interaction_term
        H = Omega/2*H_trans - 1/2*Delta*H_lgt1 + 1/4*H_lgt2 + 1/4*H_int
        return H




        
    





