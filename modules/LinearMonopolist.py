import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse, optimize
from math import sqrt
from timeit import default_timer as timer
from datetime import timedelta
from time import strftime

from modules.screening import Screening

class LinearMonopolist(Screening):
    """Create a model for 2D linear monopolist (with indivisible goods)."""
    def __init__(self, theta, f, param):
        super().__init__(f)
        self.id = f'LinearMonopolist_N{self.N}_'+strftime('%Y-%m-%d_%H-%M-%S')
        self.theta = theta; self.f = f
        self.param = param; self.substituable = param['substituable'] if 'substituable' in param.keys() else False
        self.Lambda_global = sparse.hstack((sparse.vstack([sparse.diags(theta[0,i]-theta[0,:],0, dtype=np.float32) for i in range(self.N)]),
                                     sparse.vstack([sparse.diags(theta[1,i]-theta[1,:],0, dtype=np.float32) for i in range(self.N)])),
                                     format='csr')
        self.Lambda = self.Lambda_global
        self.v0 = np.zeros(self.N**2, dtype=np.float32)
        self.y0 = self.argmax_y_lagrangian(self.v0)
        self.lamb0 = np.zeros(self.N, dtype=np.float32)
        print(f'model id = {self.id}\n')

    def all(self):
        _, s, _ = sparse.linalg.svds(self.Lambda, k=1, solver="arpack")
        self.norm_Lambda = s[0]
        # self.mask = np.full((self.N,self.N), True)
        self.indices = range(self.N**2)
        self.y = self.y0
        self.v_aug = self.v0
        self.lamb = self.lamb0
        super().super_local()
        
    def local(self, indices=None, left=True, down=True, right=False, up=False):
        if indices is None:
            n0, n1 = len(np.unique(self.theta[0])), len(np.unique(self.theta[1]))
            ind_right = [j*n0 + i for j in range(n1) for i in range(1,n0)]
            ind_top = [j*n0 + i for j in range(1,n1) for i in range(n0)]
            ind_left = [j*n0 + i for j in range(n1) for i in range(n0-1)]
            ind_bottom = [j*n0 + i for j in range(n1-1) for i in range(n0)]
            self.indices = sorted(set().union([i*self.N + i-1 for i in ind_right] if left else [],
                                              [i*self.N + i-n0 for i in ind_top] if down else [],
                                              [i*self.N + i+1 for i in ind_left] if right else [],
                                              [i*self.N + i+n0 for i in ind_bottom] if up else []))
            self.y = self.y0
            self.v_aug = np.zeros(self.N**2); self.v_aug[self.indices] = self.v0[self.indices]
            self.lamb = self.lamb0
            super().super_local()
        else:
            self.indices = indices
            super().super_local()
        
        self.Lambda = self.Lambda_global[self.indices]
        _, s, _ = sparse.linalg.svds(self.Lambda, k=1, solver="arpack")
        self.norm_Lambda = s[0]

    def ind_local(self, k):
        return sorted(set().union(*[np.arange(i, self.N**2+i, self.N+1)[:-i] for i in range(min(k,self.N))], 
                                  *[np.arange(-i, self.N**2, self.N+1)[i:] for i in range(min(k,self.N))]))
    
    def eval_S(self, y):
        return np.tile(self.f,2)*y @ self.theta.flatten()

    def grad_S(self, y):
        return np.tile(self.f,2)*self.theta.flatten()

    def prox_minusS(self, y, tau):
        if self.substituable==False:
            return np.clip(y + tau*np.tile(self.f,2)*self.theta.flatten(), 0, 1)
        else:
            z = np.clip(y + tau*np.tile(self.f,2)*self.theta.flatten(), 0, None)
            simplex_violated = np.tile(np.eye(self.N), 2)@ z > 1
            if np.any(simplex_violated):
                z0 = np.clip(1/2*(y[:self.N] - y[self.N:] + 1 + tau*(self.theta[0] -self.theta[1])), 0, 1)
                z[np.tile(simplex_violated, 2)] = np.concatenate((z0[simplex_violated], 1-z0[simplex_violated]))
            return z


    def argmax_y_lagrangian(self, v):
        return np.heaviside(np.tile(self.f,2)*self.theta.flatten() - self.Lambda.T@v, 0.5)
    
    def nlLambda(self, y): # to use nonlinear PDHG
        return self.Lambda @ y
    
    def JLambda(self, y): # to use nonlinear PDHG
        return self.Lambda