#encn304.py
from ipywidgets import*
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import (interact, fixed, interactive_output, 
    HBox, Button, VBox, Output, IntSlider, Checkbox, FloatSlider, 
    FloatLogSlider, Dropdown, FloatText, Label, Layout)

def _power_method(it,x1,x2):
    
    A = np.array([[1,2],[2,1]])
    x0 = np.array([x1,x2]).T
    x0 = x0/np.sqrt(x0.dot(x0.T))

    f,ax = plt.subplots(1,1,figsize=(6,6))
    
    x = 1.*x0

    ax.plot([0,x[0]], [0,x[1]], 'k--')
    ax.text(x[0], x[1], '$x_0$', ha='left', va='bottom')
    xi = 1.
    for i in range(it):
        xi = A.dot(x)
        x = xi/np.sqrt(xi.dot(xi))
        ev = x.T.dot(A.dot(x))/(x.T.dot(x))
        al = (i+1)/it
        ax.plot([0,x[0]], [0,x[1]], 'k-', alpha=al) 
        ax.text(x[0], x[1], '$\lambda_{:d}'.format(i+1)+'$={:4.3f}'.format(ev), 
            ha='left', va='bottom', alpha=al)
        
        deg = np.arccos(x0.T.dot(x))/np.pi*180.
        x0 = 1.*x

    ax.text(0.05,0.08,'$x_{:d}'.format(i+1)+'$=', transform=ax.transAxes, 
        ha='left', va='center')
    ax.text(0.12,0.10,'{:4.3f}'.format(x[0]), transform=ax.transAxes, 
        ha='left', va='center')
    ax.text(0.12,0.06,'{:4.3f}'.format(x[1]), transform=ax.transAxes, 
        ha='left', va='center')
    
    ax.text(0.22,0.08,r',  $\Delta\theta$'+'={:4.3f}'.format(deg)+'$^{\circ}$', transform=ax.transAxes, 
        ha='left', va='center')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax.text(1.05, 0.08, 'x=np.dot(A,x)', transform=ax.transAxes, 
        ha='left', va='center', size=8, alpha=0.2)
    ax.text(1.05, 0.06, 'x=x/np.sqrt(np.dot(x.T,x))', transform=ax.transAxes, 
        ha='left', va='center', size=8, alpha=0.2)
    ax.text(1.05, 0.04, 'ev=np.dot(np.dot(x.T,A),x)/np.dot(x.T,x)', transform=ax.transAxes, 
        ha='left', va='center', size=8, alpha=0.2)
    ax.text(1.05, 0.02, 'iteration: copy-paste, or use for loop', transform=ax.transAxes, 
        ha='left', va='center', size=8, alpha=0.2,fontstyle='italic')
def power_method():    
    it = IntSlider(1, 1, 7, 1, description='iterations')
    x1 = FloatText(1)
    x2 = FloatText(0)
    io = interactive_output(_power_method, {'it':it,'x1':x1,'x2':x2})
    
    return VBox([HBox([it,Label('$x_0$'),VBox([x1,x2])]),io])

def _earthquake_response(ti):

    f,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))

    A=np.array([[1,-1,0],[-1,3,-2],[0,-2,5]])
    ds,V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)

    t = np.linspace(0,20,101)
    x0 = np.array([-3,-2,-1])
    ci = np.dot(Vinv,x0)

    cs = ['r','g','b']
    dx,dy = [1.5,0.5]
    xs = []
    for i in range(3):
        c = cs[i]
        xi = np.sum([ci[j]*np.cos(ds[j]*t)*V[i,j] for j in range(3)],axis=0)
        ax1.plot(t,xi,c+'-',label='$x_{:d}$'.format(i+1))
        j = np.argmin(abs(t-ti))        
        ax2.plot([xi[j]-dx,xi[j]+dx,xi[j]+dx,xi[j]-dx,xi[j]-dx],
            [3-i+dy, 3-i+dy, 3-i-dy, 3-i-dy, 3-i+dy],c+'-')
        ax2.fill_between([xi[j]-dx,xi[j]+dx],[3-i-dy,3-i-dy],[3-i+dy,3-i+dy],
            color = c, alpha=0.5)
        ax1.plot(ti,xi[j],c+'o')
        xs.append(xi[j])
    ax2.plot(xs, [3, 2, 1], 'k-o')
    ax2.set_xlim([-5,5])
    ax1.legend()
    ax2.set_yticks([])
    ax2.set_xlabel('horizontal position, $x$')
    ax1.set_xlabel('time, $t$')
    ax1.set_ylabel('horizontal position, $x$')

    plt.show()


def earthquake_response():
    it = IntSlider(0, 0, 20, 1, description='time')
    io = interactive_output(_earthquake_response, {'ti':it})    
    return VBox([it,io])


def test():
    pass

def main():
    test()

if __name__ == "__main__":
    main()