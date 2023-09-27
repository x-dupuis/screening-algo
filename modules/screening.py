import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from math import sqrt
from timeit import default_timer as timer
from datetime import timedelta
from time import strftime

from modules.utils import drawArrow

class Screening:
    """
    The base class.
     
    Contains: 
        - the projection and the support function of the polyhedron K
        - the Primal-Dual Hybrid Gradient algorithm (PDHG, a.k.a Chambolle-Pock)
        - general plotting functions
    """
    def __init__(self, mu):
        self.mu = mu; self.N = mu.shape[-1]
        self.D_global = sparse.vstack([-sparse.eye(self.N, dtype=np.float32) 
                                       + sparse.coo_matrix((np.ones(self.N, dtype=np.float32), 
                                                            (np.arange(self.N, dtype=np.float32),
                                                             i*np.ones(self.N,dtype=int))), 
                                                             shape=(self.N,self.N)) 
                                                             for i in range(self.N)], format='csr')
        self.D = self.D_global; self.gamma_proj = 0.5/self.N # = 1/self.norm_D**2 with self.norm_D = sqrt(2*self.N);
        self.itmax_proj = 1e2; self.atol_proj = 1e-6; self.rtol_proj =  1e-4; self.atol_LPvalue = 1e-10; self.rtol_LPvalue = 1e-6
        
    def super_local(self):
        self.D = self.D_global[self.indices]
        _, s, _ = sparse.linalg.svds(self.D, k=1, solver="arpack") # norm_Lambda = s[0]
        self.gamma_proj = 1/s[0]**2

    def proj_K(self, w, lamb0):
        """Projection onto K by Fast Projected Gradient"""
        lamb = lamb_extra = lamb0
        it = 0; test = np.array([False,False])
        while np.all(test==False):
            it+=1; test[0]= it==self.itmax_proj
            lamb_old = lamb ## old lambda_k
            lamb = np.clip(lamb_extra - self.gamma_proj*(self.mu - self.D.T@np.clip(w - self.D@lamb_extra, 0, None)), 0, None)  ## new lambda_{k+1}
            v = np.clip(w - self.D@lamb, 0, None) ## projection of w - D*lambda onto {v >= 0}
            constraints = self.D.T@v - self.mu; obj = 1/2*np.linalg.norm(v-w)**2 + lamb@constraints; 
            test[1] = np.all(constraints  < self.rtol_proj*self.mu + self.atol_proj) and abs(lamb@constraints)< self.rtol_proj*obj + self.atol_proj
            lamb_extra = lamb + (it-1)/(it+3)*(lamb-lamb_old)   ## extrapolated lambda 
        return v, lamb, it, 0 if test[1] else np.nan #if test[0] else 0 # obj, record_objective

    def support_K(self, m):
        """
        Value of the linear program (or nan if unfeasible) = support function of the polyhedron K :
        primal: min_U <mu,U> ; DU >= m, U >= 0
        dual: max_v <m,v> ; D*v<=mu, v >= 0
        N.B. Rent(x) = LP(Lambda(x))"""
        m_aug = np.full(self.N**2,np.nan); m_aug[range(0,self.N**2,self.N)] = np.zeros(self.N); m_aug[self.indices] = m
        n = 0; test = np.array([False,False])
        U = np.zeros(self.N)
        while np.all(test==False):
            n+=1; test[0] = n==self.N
            U_old = U.copy()
            U = np.nanmax(np.split(np.concatenate(self.N*[U])+m_aug,self.N),axis=1)
            test[1]= np.linalg.norm(U-U_old)< self.atol_LPvalue + self.rtol_LPvalue*np.linalg.norm(U_old)
        return np.nan if test[0] else self.mu@U, U
    
    
    def solve(self, linear=True, stepratio=1., scale=True, warmstart=False, path=None, t_acc=1., log=1, it_max=1e4, tol_primal=1e-6, tol_dual=1e-6, ord_residual=2):
        # initialization
        t_start = timer()
        it = 0; criteria = np.array([False, False, False])
        rec_primal_residual = []; rec_dual_residual = []; rec_it_proj = []
        
        if warmstart:
            y = self.y; v = self.v_aug[self.indices]; lamb = self.lamb
        else:
            y = self.y0; v = self.v0[self.indices]; lamb = self.lamb0

        # scaling of the tolerances
        if scale:
            tol_primal = sqrt(len(y))*tol_primal; tol_dual = sqrt(len(v))*tol_dual

        tau = 1/(sqrt(stepratio)*self.norm_Lambda); sigma = sqrt(stepratio)/self.norm_Lambda # sigma (dual) = stepratio * tau (primal)
        
        prox_F = self.prox_minusS; proj_K = self.proj_K
        
        if linear:
            Lambda = self.Lambda
            Ly = Lambda@y; LTv = Lambda.T@v
        
            # loop
            while it < it_max and np.any(criteria == False):
                it += 1; y_old = y; v_old = v; LTv_old = LTv
                # primal update
                y = prox_F(y-tau*LTv,tau)
                Ly = Lambda@y
                # dual update
                y_bar = y + t_acc*(y-y_old); Ly_bar = Lambda@y_bar
                v, lamb, it_proj, charac = proj_K(v + sigma*Ly_bar,lamb) ## lamb for warm start, lamb0 for cold start
                LTv = Lambda.T@v
                # record
                rec_primal_residual.append(np.linalg.norm((y_old-y)*t_acc/tau-(LTv_old-LTv), ord=ord_residual))
                rec_dual_residual.append(np.linalg.norm((v_old-v)/sigma + (Ly_bar-Ly), ord=ord_residual))
                rec_it_proj.append(it_proj)
                # stopping criterion
                criteria = np.array([rec_primal_residual[-1] < tol_primal, rec_dual_residual[-1] < tol_dual, charac==0])
        
        else:
            Lambda = self.nlLambda; JLambda = self.JLambda
            Ly = Lambda(y); JLy = JLambda(y); LTv = JLy.T@v

             # loop
            while it < it_max and np.any(criteria == False):
                it += 1; y_old = y; v_old = v; LTv_old = LTv
                # primal update
                y = prox_F(y-tau*LTv,tau)
                Ly = Lambda(y); JLy = JLambda(y)
                # dual update
                y_bar = y + t_acc*(y-y_old); Ly_bar = Lambda(y_bar)
                v, lamb, it_proj, charac = proj_K(v + sigma*Ly_bar,lamb) ## lamb for warm start, lamb0 for cold start
                LTv = JLy.T@v
                # record
                rec_primal_residual.append(np.linalg.norm((y_old-y)*t_acc/tau-(LTv_old-LTv), ord=ord_residual))
                rec_dual_residual.append(np.linalg.norm((v_old-v)/sigma + (Ly_bar-Ly), ord=ord_residual))
                rec_it_proj.append(it_proj)
                # stopping criterion
                criteria = np.array([rec_primal_residual[-1] < tol_primal, rec_dual_residual[-1] < tol_dual, charac==0])

        # end
        v_aug = np.zeros(self.N**2); v_aug[self.indices] = v
        SLy, U = self.support_K(Ly)
        if linear:
            y_max = self.argmax_y_lagrangian(v)
            primal = self.eval_S(y) - SLy
            dual = self.eval_S(y_max) - y_max @ LTv + charac
            gap = dual - primal
        elapsed = timer() - t_start

        if log:
            print(f'convergence = {criteria.all()}, iterations = {it}, elapsed time = {str(timedelta(seconds=elapsed))}')
            print(f'primal residual = {rec_primal_residual[-1]/tol_primal:.2e} tol, dual residual = {rec_dual_residual[-1]/tol_dual:.2e} tol') 
            if linear:
                print(f'primal-dual gap = {gap:.2e}, optimal value - current value < {gap/primal:.2e} optimal value') # relative value suboptimality = (optimal value - current value) / optimal value
        
        if path is not None:
            with open(path+'.txt', 'a') as f:
                f.write(f'N = {self.N}, tau = {tau:.2e}, sigma = {sigma:.2e}, step ratio = {stepratio}, primal tol = {tol_primal:.2e}, dual tol = {tol_dual:.2e}\n')
                f.writelines(', '.join([key+f' = {value}' for key, value in self.param.items()]) + '\n')
                f.write(f'convergence = {criteria.all()}, iterations = {it}, elapsed time = {str(timedelta(seconds=elapsed))}\n')
                f.write(f'primal residual = {rec_primal_residual[-1]/tol_primal:.2e} tol, dual residual = {rec_dual_residual[-1]/tol_dual:.2e} tol\n')
                if linear:
                    f.write(f'primal-dual gap = {gap:.2e}, relative suboptimality < {gap/primal:.2e}\n')
                f.write('\n')

        # model update
        self.y = y; self.v = v; self.lamb = lamb; self.v_aug = v_aug; self.U = U
        self.it = it; self.elapsed = elapsed
        self.rec_primal_residual = np.array(rec_primal_residual); self.rec_dual_residual = np.array(rec_dual_residual); self.rec_it_proj = np.array(rec_it_proj)
        self.IR_binding = np.flatnonzero((self.D.T@v - self.mu) < -(self.rtol_proj*self.mu + self.atol_proj))
        self.IC_binding = np.argwhere(v_aug.reshape(self.N, self.N))
        if linear:
            self.primal = primal; self.gap = gap
            self.IC_violated = np.flatnonzero(self.Lambda_global @ y - self.D_global @ U > self.rtol_proj + self.atol_proj)



    def residuals(self, title=None, path=None, **fig_kw):
        self.fig, ax = plt.subplots(**fig_kw)
        _ = ax.semilogy(self.rec_primal_residual, label='primal')
        _ = ax.semilogy(self.rec_dual_residual, label='dual')
        _ = ax.legend()
        _ = ax.set_title(title)
        if path is not None:
            self.fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


    def range(self, figsize=(5,5), s=20, title=None, path=None, **kwargs): # use aspect='equal'
        self.fig, ax = plt.subplots(1,1 ,figsize=figsize, subplot_kw=kwargs) # subplot_kw=dict(aspect='equal',) 
        _ = ax.scatter(self.y[:self.N], self.y[self.N:],color='tab:blue', s=s,label='products')
        _ = ax.set_xlabel(r'$y_1$'); _ = ax.set_ylabel(r'$y_2$')
        _ = ax.set_title(title)
        if path is not None:
            self.fig.savefig(path, bbox_inches="tight", pad_inches=0.05)

    def constraints(self, figsize=(5,5), s=20, title=None, path=None, **kwargs): # use aspect='equal'
        IC = 'IC' if title else 'IC binding'; IR = 'IR' if title else 'IR binding'
        self.fig, ax = plt.subplots(1,1 ,figsize=figsize, subplot_kw=kwargs) # subplot_kw=dict(aspect='equal',) 
        _ = ax.scatter(self.theta[0],self.theta[1],facecolors='w',edgecolors='k',s=s, zorder=2.5)
        _ = ax.scatter([],[], marker='>', c='k', label=IC)
        _ = ax.scatter(self.theta[0][self.IR_binding],self.theta[1][self.IR_binding],label=IR,c='tab:green',s=s,zorder=2.5)
        for i,j in self.IC_binding:
            _ = drawArrow(ax,self.theta[0,i],self.theta[0,j],self.theta[1,i],self.theta[1,j])
        _ = ax.set_xlabel(r'$\theta_1$'); _ = ax.set_ylabel(r'$\theta_2$')
        
        if title is None:
            _ = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncols=2, mode="expand", borderaxespad=0.)
        else:
            _ = ax.set_title(title)
            _ = ax.legend(bbox_to_anchor=(1.02, 1.), loc='lower right')
            
        if path is not None:
            self.fig.savefig(path, bbox_inches="tight", pad_inches=0.05)

    def products_constraints(self, figsize=(10,5), s=20, title=None, path=None):
        IC = 'IC' if title else 'IC binding'; IR = 'IR' if title else 'IR binding'
        self.fig, axs = plt.subplots(1,2, figsize=(10,5) ,subplot_kw=dict(aspect='equal',))
        _ = axs[0].scatter(self.y[:self.N], self.y[self.N:],color='tab:blue',s=s,label='products')
        _ = axs[0].set_xlabel(r'$y_1$'); _ = axs[0].set_ylabel(r'$y_2$')
        _ = axs[1].scatter(self.theta[0],self.theta[1],facecolors='w',edgecolors='k',s=s, zorder=2.5)
        _ = axs[1].scatter([],[], marker='>', c='k', label=IC)
        _ = axs[1].scatter(self.theta[0][self.IR_binding],self.theta[1][self.IR_binding],label=IR,c='tab:green',s=s,zorder=2.5)
        for i,j in self.IC_binding:
            _ = drawArrow(axs[1],self.theta[0,i],self.theta[0,j],self.theta[1,i],self.theta[1,j])
        _ = axs[1].set_xlabel(r'$\theta_1$'); _ = axs[1].set_ylabel(r'$\theta_2$')
        _ = axs[1].legend(bbox_to_anchor=(1.02,1), loc='lower right')
        
        if title is None:
            _ = axs[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncols=2, mode="expand", borderaxespad=0.)
        else:
            _ = axs[0].set_title(title[0])
            _ = axs[1].set_title(title[1])
            _ = axs[1].legend(bbox_to_anchor=(1.02, 1.), loc='lower right')
            
        if path is not None:
            self.fig.savefig(path, bbox_inches="tight", pad_inches=0.05)