#encn304.py
from ipywidgets import*
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from scipy.optimize import root

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

    f,(ax1,ax2) = plt.subplots(1,2,figsize=(12,5))

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

def _euler_method(step, h):
    
    f,(ax, ax2) = plt.subplots(1,2, figsize=(12,5))

    # initialise ODE
    x = [0,]
    y = [1,]
    h0 = 0.1
    
    # setup axes limits
    xlim = np.array([-0.05,1.15])
    ylim = [-0.9,10]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    def dydx(x,y): 
        return (1.+x*y)**2

    for i in range(int(step)):
        y.append(y[-1]+h0*dydx(x[-1], y[-1]))
        x.append(x[-1]+h0)		
        
    if abs(step-int(step))>0.25:
        # plot derivative
        dydx0 = dydx(x[-1], y[-1])
        ys = dydx0*(xlim - x[-1])+y[-1]
        ax.plot(xlim, ys, 'r--')		
        ax.text(0.95*xlim[-1], np.min([1.05*ys[-1],9.]), 'compute derivative: $f^{'+'({:d})'.format(int(step))+'}=(t^{'+'({:d})'.format(int(step))+'},x^{'+'({:d})'.format(int(step))+'})$', ha = 'right', va = 'bottom', color = 'r')
    else:	
        dy = 0.4
        dx = 0.04
        ax.arrow(x[-2], y[-2]-dy, h0, 0, length_includes_head = True, head_width = 0.2, head_length = 0.02, color= 'r', linewidth = 0.5)
        ax.arrow(x[-1], y[-2]-dy, -h0, 0, length_includes_head = True, head_width = 0.2, head_length = 0.02, color= 'r', linewidth = 0.5)
        ax.text(0.5*(x[-1]+x[-2]), y[-2]-2*dy, '$t^{'+'({:d})'.format(int(step))+'}=t^{'+'({:d})'.format(int(step-1))+'}+\Delta t$', ha = 'center', va = 'top', color = 'r')
        
        ax.arrow(x[-1]+dx, y[-2], 0, y[-1]-y[-2], length_includes_head = True, head_width = 0.02, head_length = 0.2, color= 'r', linewidth = 0.5)
        ax.arrow(x[-1]+dx, y[-1], 0, -y[-1]+y[-2], length_includes_head = True, head_width = 0.02, head_length = 0.2, color= 'r', linewidth = 0.5)
        
        ax.text(x[-1]+2*dx, 0.5*(y[-1]+y[-2]), 'take step: $x^{'+'({:d})'.format(int(step))+'}=x^{'+'({:d})'.format(int(step-1))+'}+\Delta t\,f^{'+'({:d})'.format(int(step-1))+'}$', ha = 'left', va = 'center', color = 'r')
                
    ax.plot(x,y,'ko-', mfc = 'k')
    
    ax.plot(x[-1],y[-1],'ko', mfc = 'w')
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    
    # second plot, effect of step size
    x = [0,]
    y = [1,]
    x0 = [0,]
    y0 = [1,]
    
    while x[-1] < 1.:
        y.append(y[-1]+h*dydx(x[-1], y[-1]))
        x.append(x[-1]+h)	
    while x0[-1] < 1.:
        y0.append(y0[-1]+h0*dydx(x0[-1], y0[-1]))
        x0.append(x0[-1]+h0)	

    y0 = y0[:-1]
    x0 = x0[:-1]
    
    ax2.plot(x,y,'ko-', mfc = 'k', label = 'h={:3.2f}'.format(h))
    ax2.plot(x0,y0,'ko-', mfc = 'k', alpha = 0.5, label = 'h={:3.2f}'.format(h0))
    
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y(x)$')
    ax2.set_ylim([0,20])
    ax2.set_xlim(xlim)
    
    ax2.legend(loc=2)
    plt.show()
def euler_method():
    
    steps = FloatSlider(value=0.5, min=0.5, max=10, step=0.5, description='steps')
    h = FloatSlider(value=0.1, min=0.02, max=0.2, step=0.02, description='h')
    io = interactive_output(_euler_method, {'step':steps,'h':h})    
    return VBox([HBox([steps, h]),io])

def _euler_error(steps, predict_value):
        
    f,ax = plt.subplots(1,1, figsize=[12, 5])
    p = [8, 8.5]
    x = np.linspace(0,10., 1001)
    
    ax.set_xlim([0,10])
    ax.plot([0,10],[0,0],'k:')
    
    def dvarsin(x, *p): 
        return np.sin(p[0]*np.sin(x)*np.sqrt(x)+np.cos(p[1]*x)/(x+1))

    xs = np.linspace(0, predict_value,10*steps)
    h = xs[1]-xs[0]
    ya = 0.*xs
    for i in range(len(xs)-1):
        ya[i+1] = ya[i] + h/2*(dvarsin(xs[i], *p)+dvarsin(xs[i+1], *p))
        
    ax.set_xlabel('time, $t$')
    ax.set_ylabel('solution, $x$')
    
    # plot Euler steps
    h = predict_value/steps
    xs = np.arange(steps+1)*h
    ys = 0.*xs
    for i in range(steps):
        ys[i+1] = ys[i] + h*dvarsin(xs[i], *p)
        
    ax.plot(xs,ys, '.b-', label = 'Euler')
    
    # plot error bar
    xest = xs[-1]
    yest = ys[-1]
    ytrue = ya[-1]
    
    ax.plot([xest, xest], [yest, ytrue], 'r-', lw = 2, label = 'error')
    ymid = 0.5*(yest+ytrue)
    err = abs((yest-ytrue)/ytrue)*100
    if err < 1.0:
        wgt = 'bold'
        err_str = ' err < 1%'
    else:
        wgt = 'normal'
        err_str = ' err = {:d}%'.format(int(err))
    
    ax.text(xest, ymid, err_str, color = 'r', fontweight = wgt)        
    ax.legend(loc = 4)
def euler_error():
    box1 = IntText(value = 20, description='with steps')
    box2 = BoundedFloatText(value = 2.2, description='predict at')
    io = interactive_output(_euler_error, {'steps':box1,'predict_value':box2})    
    return VBox([HBox([box2, box1]),io])

def root_equation(yk1, yk, h, xk, f, *p):
    return yk - yk1 + h*f(xk+h, yk1, *p) 
    # implement backward Euler method
def _euler_stability(method,step):
    # create figure
    f,ax = plt.subplots(1,1)
    f.set_size_inches([12,5])

    def dydx2(x,y): return -10*y
    
    x0,x1 = [0,1]
    y0 = 1
    
    h = x1/step
    
    if method == 'Euler':
        x = [x0,]
        y = [y0,]
        while x[-1] < x1:		
            y.append(y[-1]+h*dydx2(x[-1],y[-1]))
            x.append(x[-1]+h)
            
        ax.plot(x,y,'b--x', label='Euler')

    elif method == 'Backward Euler':
        x = [x0,]
        y = [y0,]
        while x[-1] < x1:
            ynew = root(root_equation, y[-1], args = (y[-1], h, x[-1], dydx2))
            y.append(ynew.x)
            x.append(x[-1]+h)
            
        ax.plot(x,y,'r--x', label='Backward Euler')
    
    elif method == 'Improved Euler':
        x = [x0,]
        y = [y0,]
        while x[-1] < x1:		
            y.append(y[-1]+h/2.*(dydx2(x[-1],y[-1])+dydx2(x[-1]+h,y[-1]+h*dydx2(x[-1],y[-1]))))
            x.append(x[-1]+h)
            
        ax.plot(x,y,'g--x', label='Improved Euler')
    
    xv = np.linspace(x0,y0,101)
    yv = np.exp(-10*xv)
    ax.plot(xv,yv,'c-', lw=2, label='exact')	
    ax.legend(loc=1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlim([x0,x1])
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    
    ax.text(0.5, 0.95, 'a=-10, $\Delta t$={:4.3f}'.format(h), transform=ax.transAxes, ha = 'center', va = 'top')
    plt.show()
    
def euler_stability():
    steps = IntSlider(15, 3, 15, 1, description='steps')
    method = Dropdown(
    options=['Euler', 'Improved Euler', 'Backward Euler'],
    value='Euler',
    description='method')
    io = interactive_output(_euler_stability, {'method':method,'step':steps})    
    return VBox([HBox([steps, method]),io])

def test():
    pass

def main():
    test()

if __name__ == "__main__":
    main()