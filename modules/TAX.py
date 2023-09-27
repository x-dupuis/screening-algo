import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse, optimize
from math import sqrt
from timeit import default_timer as timer
from datetime import timedelta
from time import strftime

from modules.solver import PDHG

class TAX(PDHG):
    """Create a model for 2D taxation."""
    def __init__(self, theta, f, param):
        super().__init__(param['lambd']*f)
        self.id = f'TAX_N{self.N}_'+strftime('%Y-%m-%d_%H-%M-%S')
        self.theta = theta; self.f = f
        self.param = param; self.constrained = param['constrained'] if 'constrained' in param.keys() else False
        self.eta = param['eta']; self.R = param['R']; self.w = param['w']; 
        
        ## exponential utility function
        self.u = lambda c : (1 - np.exp(-self.eta*c))/self.eta
        self.du = lambda c : np.exp(-self.eta*c)

        self.Lambda2 = sparse.vstack([sparse.diags(theta[1,:]-theta[1,i],0, dtype=np.float32) for i in range(self.N)], format='csr')
        self.y0 = np.concatenate((self.theta[0],np.ones(self.N)))
        self.v0 = np.zeros(self.N**2, dtype=np.float32)
        self.lamb0 = np.zeros(self.N, dtype=np.float32)
        print(f'model id = {self.id}')

    def all(self, seed=None, under_estim=False):
        rng = np.random.default_rng(seed)
        if seed is None:
            y = self.y0
        else:
            y = np.concatenate((rng.random(self.N)*self.theta[0],rng.random(self.N)))
        if under_estim:
            w = rng.normal(size=2*self.N)
            self.norm_Lambda = sqrt(np.linalg.norm(self.JLambda(y).T @ self.JLambda(y) @ w) / np.linalg.norm(w))
        else:
            _, s, _ = sparse.linalg.svds(self.JLambda(y), k=1, solver="arpack")
            self.norm_Lambda = s[0]
        self.indices = range(self.N**2)
        self.y = self.y0
        self.v_aug = self.v0
        self.lamb = self.lamb0
        super().super_local()

    def nlLambda(self, y):
        return np.concatenate([self.u(self.theta[0,i]-y[:self.N]) - self.u(self.theta[0,:]-y[:self.N]) for i in range(self.N)], dtype=np.float32) + self.Lambda2@y[self.N:]
    
    def JLambda(self, y):
        return  sparse.hstack((sparse.vstack([sparse.diags(self.du(self.theta[0,:]-y[:self.N]) - self.du(self.theta[0,i]-y[:self.N]),0, dtype=np.float32)
                                               for i in range(self.N)]), self.Lambda2), format='csr')

    def prox_H(self,s,l,e,x,tau):
        """Proximal operator of H(s,l) = -u(e-s) - Rs - (w-x)l, 0<=s<=e, 0<=l<=1"""
        equation = lambda r : self.du(e-r) - self.R + (r-s)/tau
        if self.constrained:
            if equation(0) >= 0:
                r = 0
            elif equation(e) <= 0:
                r = e
            else:
                r = optimize.root_scalar(equation, bracket=[0,e]).root
        else:
            # if equation(0) >= 0:
            #     r = 0
            # else:
            #     r = optimize.root_scalar(equation, x0=0, x1=e).root
            r = optimize.root_scalar(equation, x0=0, x1=e).root
        return r, np.clip(l+tau*(self.w-x),0,1)
    
    def prox_minusS(self,z,tau):
        """Proximal operator of -S(y) = sum_i f_i H(y_i, theta_i)"""
        y = np.empty(2*self.N)
        for i in range(self.N):
            y[i], y[i+self.N] = self.prox_H(z[i],z[i+self.N],self.theta[0,i],self.theta[1,i],tau*self.f[i])
        return y
    
    def eval_S(self,y):
        return np.sum(self.f * (self.u(self.theta[0] - y[:self.N]) + self.R*y[:self.N] + (self.x - self.theta[1])*y[N:]))
       
    def output(self, path=None):
        self.c = self.theta[0] - self.y[:self.N]
        self.tau = self.R - self.du(self.theta[0] - self.y[:self.N])
        self.T = self.u(self.theta[0] - self.y[:self.N]) + self.R*self.y[:self.N] + (self.w - self.theta[1])*self.y[self.N:] - self.U
        self.c2 = self.U - self.u(self.theta[0] - self.y[:self.N]) + self.theta[1]*self.y[self.N:]
        
        self.df_output = pd.DataFrame({'e': self.theta[0],
                                       'x': self.theta[1],
                                       's': self.y[:self.N],
                                       'c': self.c,
                                       'l': self.y[self.N:],
                                       'tau': self.tau,
                                       'U': self.U,
                                       'T': self.T,
                                       'c2': self.c2,})
        with pd.option_context(#'display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,):
            print(self.df_output)
        
        if path is not None:
            df_param = pd.DataFrame.from_dict(self.param.items())
            with pd.ExcelWriter(path+'.xlsx') as writer:  
                self.df_output.round(3).to_excel(writer, sheet_name='output')
                df_param.to_excel(writer, sheet_name='parameters')

    
    def display(self, variable, title=None, label=None, path=None, s=20, figsize=(5,5),**kwargs):
        self.fig, ax = plt.subplots(1,1 ,figsize=figsize, subplot_kw=kwargs) #subplot_kw=dict(aspect='equal',)
        _ = ax.set_xlabel(r'initial endowment $e$'); _ = ax.set_ylabel(r'disutility of working $x$')
        _ = ax.set_title(title)
        scatter = ax.scatter(self.theta[0], self.theta[1], c=variable)
        _ = self.fig.colorbar(scatter, label=label)

        if path is not None:
            self.fig.savefig(path, bbox_inches="tight", pad_inches=0.05)