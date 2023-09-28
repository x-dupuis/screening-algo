import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from math import sqrt, ceil
from timeit import default_timer as timer
from datetime import timedelta
import sys
sys.path.append('..')
from modules.TAX import TAX

#############

### SETTING

## Types setting
n0, n1 = 10, 10
theta0, theta1 =  np.linspace(1,3,num=n0), np.linspace(0,1,num=n1)  #  [1, 1.5, 3], [0, 0.5, 1]
theta0, theta1 = np.meshgrid(theta0,theta1)
theta = np.stack((theta0.flatten(), theta1.flatten())); N = theta.shape[-1]     # number of types

f = np.ones(N).flatten()    # weights of distribution

## Model parameters setting
param = {'lambd':0.5, 'R':1., 'w':1, 'eta':1, 'constrained':False} # 'constrained': boolean for the constraint 0 <= s <= e

## Objects
model = TAX(theta, f, param)

### DIRECT RESOLUTION (with all the constraints)

model.all(seed=1, under_estim=True)
model.solve(linear=False, stepratio=sqrt(N), it_max=1e5,) #path='results/'+model.id)
model.residuals(title='residuals')
# # model.constraints(path='results/'+model.id+'_constraints')

### OUTPUT

model.output(path='results/'+model.id)
model.display(model.y[N:], label=r'labor supply $l$', path='results/'+model.id+'_labor') #aspect='equal'
model.display(model.tau, label=r'marginal tax rate $\tau$', path='results/'+model.id+'_tau')

print('\n')
plt.show()