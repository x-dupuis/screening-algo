import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from math import sqrt, ceil
from timeit import default_timer as timer
from datetime import timedelta
import sys
sys.path.append('..')
from modules import PAP

############# MULTIPRODUCT MONOPOLIST ############

### SETTING 1

## Types setting
n0, n1 = 5, 5
theta0, theta1 =  np.linspace(1,2,num=n0, dtype=np.float32), np.linspace(1,2,num=n1, dtype=np.float32)  #  [1, 1.5, 3], [0, 0.5, 1]
theta0, theta1 = np.meshgrid(theta0,theta1)
theta = np.stack((theta0.flatten(), theta1.flatten())); N = theta.shape[-1]     # number of types

f = np.ones(N, dtype=np.float32).flatten()    # weights of the distribution

## Model parameters setting
param = {'constrained':False}   # boolean for the constraint y >= 0

## Objects
model = PAP(theta, f, param)

### DIRECT RESOLUTION (with all the constraints)

model.all()
model.solve(stepratio=N/2, it_max=1e4, tol_primal=1e-6, tol_dual=1e-6, scale=True,) #path='results/'+model.id
model.residuals(title='residuals')
model.range(title='Product range',) # path='results/'+model.id+'_products')
model.constraints() #path='results/'+model.id+'_constraints')

print('\n')
plt.show()

#############

### SETTING 2

## Types setting
n0, n1 = 20, 20
theta0, theta1 =  np.linspace(1,2,num=n0, dtype=np.float32), np.linspace(1,2,num=n1, dtype=np.float32)  #  [1, 1.5, 3], [0, 0.5, 1]
theta0, theta1 = np.meshgrid(theta0,theta1)
theta = np.stack((theta0.flatten(), theta1.flatten())); N = theta.shape[-1]     # number of types

f = np.ones(N, dtype=np.float32).flatten()    # weights of the distribution

## Model parameters setting
param = {'constrained':False}   # boolean for the constraint y >= 0

## Objects
model = PAP(theta, f, param)

### ADAPTATIVE APPROACH (starting with the local constraints)

kmax = ceil(sqrt(N))+1
t_start = timer()
model.local(); k = 0; test = np.array([False,False])
while np.all(test==False):
    k += 1
    print(f'OPTIMIZATION {k}: {len(model.indices)} IC constraints')
    model.solve(warmstart=True, it_max = 1e5, stepratio=len(model.indices),) # path='results/'+model.id) #sqrt(len(model.indices))
    # model.residuals()
    print(f'IC constraints: {len(model.IC_binding)} binding, {len(model.IC_violated)} violated\n')
    # model.indices = sorted(set(model.indices).union(model.IC_violated))
    model.indices = sorted(set(model.indices).union(set(model.IC_violated).intersection(model.ind_local(k*ceil(sqrt(N))))))
    model.local(model.indices)
    test = np.array([k==kmax, len(model.IC_violated)==0])
elapsed = timer() - t_start
print(f'elapsed time = {str(timedelta(seconds=elapsed))}')

model.range(title='Product range', s=10, ) # path='results/'+model.id)
# model.constraints()

print('\n')
plt.show()