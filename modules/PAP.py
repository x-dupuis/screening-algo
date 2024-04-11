import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse, optimize
from math import sqrt
from timeit import default_timer as timer
from datetime import timedelta
from time import strftime

from modules.screening import Screening

class PAP(Screening):
    """Create a model for 2D monopolist."""
    def __init__(self, theta, f, param):
        super().__init__(f)
        self.id = f'PAP_N{self.N}_'+strftime('%Y-%m-%d_%H-%M-%S')
        self.theta = theta; self.f = f
        self.param = param; self.constrained = param['constrained'] if 'constrained' in param.keys() else False
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
        return np.tile(self.f,2)*y@(self.theta.flatten() - 1/2*y)

    def grad_S(self, y):
        return np.tile(self.f,2)*(self.theta.flatten()- y)

    def prox_minusS(self, y, tau):
        return np.clip((y + tau*np.tile(self.f,2)*self.theta.flatten())/(1+tau*np.tile(self.f,2)),0,None) if self.constrained else (y + tau*np.tile(self.f,2)*self.theta.flatten())/(1+tau*np.tile(self.f,2))

    def argmax_y_lagrangian(self, v):
        return np.clip(np.concatenate((self.theta[0] - (self.Lambda.T@v)[:self.N]/self.f, self.theta[1] - (self.Lambda.T@v)[self.N:]/self.f)),0,None) if self.constrained else np.concatenate((self.theta[0] - (self.Lambda.T@v)[:self.N]/self.f, self.theta[1] - (self.Lambda.T@v)[self.N:]/self.f))
    
    def nlLambda(self, y): # to use nonlinear PDHG
        return self.Lambda @ y
    
    def JLambda(self, y): # to use nonlinear lPDHG
        return self.Lambda
    
    def output(self, path=None):
        self.p = self.theta[0] * self.y[:self.N] + self.theta[1] * self.y[self.N:] - self.U 

        self.df_output = pd.DataFrame({'theta1': self.theta[0],
                                       'theta2': self.theta[1],
                                       'f': self.f,
                                       'y1': self.y[:self.N],
                                       'y2': self.y[self.N:],
                                       'U': self.U,
                                       'p': self.p,})
        with pd.option_context(#'display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,):
            print(self.df_output)
        
        if path is not None:
            df_param = pd.DataFrame.from_dict(self.param.items())
            with pd.ExcelWriter(path+'.xlsx') as writer:  
                self.df_output.round(3).to_excel(writer, sheet_name='output')
                df_param.to_excel(writer, sheet_name='parameters')

    
    def display(self, variable, title=None, label=None, path=None, s=20, figsize=(5,5), cmap=None, **kwargs):
        self.fig, ax = plt.subplots(1,1 ,figsize=figsize, subplot_kw=kwargs) #subplot_kw=dict(aspect='equal',)
        _ = ax.set_xlabel(r'$\theta_1$'); _ = ax.set_ylabel(r'$\theta_2$')
        _ = ax.set_title(title)
        scatter = ax.scatter(self.theta[0], self.theta[1], c=variable, cmap=cmap)
        _ = self.fig.colorbar(scatter, label=label)

        if path is not None:
            self.fig.savefig(path, bbox_inches="tight", pad_inches=0.05)