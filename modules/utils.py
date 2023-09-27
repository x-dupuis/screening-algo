import numpy as np
import matplotlib.pyplot as plt
        
def drawArrow(ax,xA,xB,yA,yB,c='k',ls='-'):
    n = 50; x  = np.linspace(xA,xB,2*n+1); y  = np.linspace(yA,yB,2*n+1);
    ax.plot(x,y,color=c,linestyle=ls)
    ax.annotate("", xy=(x[n],y[n]), xytext=(x[n-1],y[n-1]), arrowprops=dict(arrowstyle="-|>", color=c) , size=15, zorder=2)