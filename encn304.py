#encn304.py
from ipywidgets import*
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from scipy.optimize import root

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from scipy.integrate import trapz
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


TEXTSIZE = 14

def _power_method(it,x1,x2):
    
    A=np.array([[1,2],[2,1]])
    x0=np.array([x1,x2]).T
    x0=x0/np.sqrt(x0.dot(x0.T))

    f,ax=plt.subplots(1,1,figsize=(6,6))
    
    x=1.*x0

    ax.plot([0,x[0]], [0,x[1]], 'k--')
    ax.text(x[0], x[1], '$x_0$', ha='left', va='bottom')
    xi=1.
    for i in range(it):
        xi=A.dot(x)
        x=xi/np.sqrt(xi.dot(xi))
        ev=x.T.dot(A.dot(x))/(x.T.dot(x))
        al=(i+1)/it
        ax.plot([0,x[0]], [0,x[1]], 'k-', alpha=al) 
        ax.text(x[0], x[1], '$\lambda_{:d}'.format(i+1)+'$={:4.3f}'.format(ev), 
            ha='left', va='bottom', alpha=al)
        
        deg=np.arccos(x0.T.dot(x))/np.pi*180.
        x0=1.*x

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
    it=IntSlider(1, 1, 7, 1, description='iterations')
    x1=FloatText(1)
    x2=FloatText(0)
    io=interactive_output(_power_method, {'it':it,'x1':x1,'x2':x2})
    
    return VBox([HBox([it,Label('$x_0$'),VBox([x1,x2])]),io])

def _earthquake_response(ti):

    f,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))

    A=np.array([[1,-1,0],[-1,3,-2],[0,-2,5]])
    ds,V=np.linalg.eig(A)
    Vinv=np.linalg.inv(V)

    t=np.linspace(0,20,101)
    x0=np.array([-3,-2,-1])
    ci=np.dot(Vinv,x0)

    cs=['r','g','b']
    dx,dy=[1.5,0.5]
    xs=[]
    for i in range(3):
        c=cs[i]
        xi=np.sum([ci[j]*np.cos(ds[j]*t)*V[i,j] for j in range(3)],axis=0)
        ax1.plot(t,xi,c+'-',label='$x_{:d}$'.format(i+1))
        j=np.argmin(abs(t-ti))        
        ax2.plot([xi[j]-dx,xi[j]+dx,xi[j]+dx,xi[j]-dx,xi[j]-dx],
            [3-i+dy, 3-i+dy, 3-i-dy, 3-i-dy, 3-i+dy],c+'-')
        ax2.fill_between([xi[j]-dx,xi[j]+dx],[3-i-dy,3-i-dy],[3-i+dy,3-i+dy],
            color=c, alpha=0.5)
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
    it=IntSlider(0, 0, 20, 1, description='time')
    io=interactive_output(_earthquake_response, {'ti':it})    
    return VBox([it,io])

def _euler_method(step, h):
    
    f,(ax, ax2)=plt.subplots(1,2, figsize=(12,5))

    # initialise ODE
    x=[0,]
    y=[1,]
    h0=0.1
    
    # setup axes limits
    xlim=np.array([-0.05,1.15])
    ylim=[-0.9,10]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    def dydx(x,y): 
        return (1.+x*y)**2

    for i in range(int(step)):
        y.append(y[-1]+h0*dydx(x[-1], y[-1]))
        x.append(x[-1]+h0)		
        
    if abs(step-int(step))>0.25:
        # plot derivative
        dydx0=dydx(x[-1], y[-1])
        ys=dydx0*(xlim - x[-1])+y[-1]
        ax.plot(xlim, ys, 'r--')		
        ax.text(0.95*xlim[-1], np.min([1.05*ys[-1],9.]), 'compute derivative: $f^{'+'({:d})'.format(int(step))+'}=(t^{'+'({:d})'.format(int(step))+'},x^{'+'({:d})'.format(int(step))+'})$', ha='right', va='bottom', color='r')
    else:	
        dy=0.4
        dx=0.04
        ax.arrow(x[-2], y[-2]-dy, h0, 0, length_includes_head=True, head_width=0.2, head_length=0.02, color= 'r', linewidth=0.5)
        ax.arrow(x[-1], y[-2]-dy, -h0, 0, length_includes_head=True, head_width=0.2, head_length=0.02, color= 'r', linewidth=0.5)
        ax.text(0.5*(x[-1]+x[-2]), y[-2]-2*dy, '$t^{'+'({:d})'.format(int(step))+'}=t^{'+'({:d})'.format(int(step-1))+'}+\Delta t$', ha='center', va='top', color='r')
        
        ax.arrow(x[-1]+dx, y[-2], 0, y[-1]-y[-2], length_includes_head=True, head_width=0.02, head_length=0.2, color= 'r', linewidth=0.5)
        ax.arrow(x[-1]+dx, y[-1], 0, -y[-1]+y[-2], length_includes_head=True, head_width=0.02, head_length=0.2, color= 'r', linewidth=0.5)
        
        ax.text(x[-1]+2*dx, 0.5*(y[-1]+y[-2]), 'take step: $x^{'+'({:d})'.format(int(step))+'}=x^{'+'({:d})'.format(int(step-1))+'}+\Delta t\,f^{'+'({:d})'.format(int(step-1))+'}$', ha='left', va='center', color='r')
                
    ax.plot(x,y,'ko-', mfc='k')
    
    ax.plot(x[-1],y[-1],'ko', mfc='w')
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    
    # second plot, effect of step size
    x=[0,]
    y=[1,]
    x0=[0,]
    y0=[1,]
    
    while x[-1] < 1.:
        y.append(y[-1]+h*dydx(x[-1], y[-1]))
        x.append(x[-1]+h)	
    while x0[-1] < 1.:
        y0.append(y0[-1]+h0*dydx(x0[-1], y0[-1]))
        x0.append(x0[-1]+h0)	

    y0=y0[:-1]
    x0=x0[:-1]
    
    ax2.plot(x,y,'ko-', mfc='k', label='h={:3.2f}'.format(h))
    ax2.plot(x0,y0,'ko-', mfc='k', alpha=0.5, label='h={:3.2f}'.format(h0))
    
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y(x)$')
    ax2.set_ylim([0,20])
    ax2.set_xlim(xlim)
    
    ax2.legend(loc=2)
    plt.show()
def euler_method():
    
    steps=FloatSlider(value=0.5, min=0.5, max=10, step=0.5, description='steps')
    h=FloatSlider(value=0.1, min=0.02, max=0.2, step=0.02, description='h')
    io=interactive_output(_euler_method, {'step':steps,'h':h})    
    return VBox([HBox([steps, h]),io])

def _euler_error(steps, predict_value):
        
    f,ax=plt.subplots(1,1, figsize=[12, 5])
    p=[8, 8.5]
    x=np.linspace(0,10., 1001)
    
    ax.set_xlim([0,10])
    ax.plot([0,10],[0,0],'k:')
    
    def dvarsin(x, *p): 
        return np.sin(p[0]*np.sin(x)*np.sqrt(x)+np.cos(p[1]*x)/(x+1))

    xs=np.linspace(0, predict_value,10*steps)
    h=xs[1]-xs[0]
    ya=0.*xs
    for i in range(len(xs)-1):
        ya[i+1]=ya[i] + h/2*(dvarsin(xs[i], *p)+dvarsin(xs[i+1], *p))
        
    ax.set_xlabel('time, $t$')
    ax.set_ylabel('solution, $x$')
    
    # plot Euler steps
    h=predict_value/steps
    xs=np.arange(steps+1)*h
    ys=0.*xs
    for i in range(steps):
        ys[i+1]=ys[i] + h*dvarsin(xs[i], *p)
        
    ax.plot(xs,ys, '.b-', label='Euler')
    
    # plot error bar
    xest=xs[-1]
    yest=ys[-1]
    ytrue=ya[-1]
    
    ax.plot([xest, xest], [yest, ytrue], 'r-', lw=2, label='error')
    ymid=0.5*(yest+ytrue)
    err=abs((yest-ytrue)/ytrue)*100
    if err < 1.0:
        wgt='bold'
        err_str=' err < 1%'
    else:
        wgt='normal'
        err_str=' err={:d}%'.format(int(err))
    
    ax.text(xest, ymid, err_str, color='r', fontweight=wgt)        
    ax.legend(loc=4)
def euler_error():
    box1=IntText(value=20, description='with steps')
    box2=BoundedFloatText(value=2.2, description='predict at')
    io=interactive_output(_euler_error, {'steps':box1,'predict_value':box2})    
    return VBox([HBox([box2, box1]),io])

def root_equation(yk1, yk, h, xk, f, *p):
    return yk - yk1 + h*f(xk+h, yk1, *p) 
    # implement backward Euler method
def _euler_stability(method,step):
    # create figure
    f,ax=plt.subplots(1,1)
    f.set_size_inches([12,5])

    def dydx2(x,y): return -10*y
    
    x0,x1=[0,1]
    y0=1
    
    h=x1/step
    
    if method == 'Euler':
        x=[x0,]
        y=[y0,]
        while x[-1] < x1:		
            y.append(y[-1]+h*dydx2(x[-1],y[-1]))
            x.append(x[-1]+h)
            
        ax.plot(x,y,'b--x', label='Euler')

    elif method == 'Backward Euler':
        x=[x0,]
        y=[y0,]
        while x[-1] < x1:
            ynew=root(root_equation, y[-1], args=(y[-1], h, x[-1], dydx2))
            y.append(ynew.x)
            x.append(x[-1]+h)
            
        ax.plot(x,y,'r--x', label='Backward Euler')
    
    elif method == 'Improved Euler':
        x=[x0,]
        y=[y0,]
        while x[-1] < x1:		
            y.append(y[-1]+h/2.*(dydx2(x[-1],y[-1])+dydx2(x[-1]+h,y[-1]+h*dydx2(x[-1],y[-1]))))
            x.append(x[-1]+h)
            
        ax.plot(x,y,'g--x', label='Improved Euler')
    
    xv=np.linspace(x0,y0,101)
    yv=np.exp(-10*xv)
    ax.plot(xv,yv,'c-', lw=2, label='exact')	
    ax.legend(loc=1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlim([x0,x1])
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    
    ax.text(0.5, 0.95, '$\lambda$=-10, $\Delta t$={:4.3f}'.format(h), transform=ax.transAxes, ha='center', va='top')
    plt.show()
def euler_stability():
    steps=IntSlider(15, 3, 15, 1, description='steps')
    method=Dropdown(
    options=['Euler', 'Improved Euler', 'Backward Euler'],
    value='Euler',
    description='method')
    io=interactive_output(_euler_stability, {'method':method,'step':steps})    
    return VBox([HBox([steps, method]),io])

# function parameters
def root_equation(yk1, yk, h, xk, f, *p):
    return yk - yk1 + h*f(xk+h, yk1, *p) 
    # implement backward Euler method
def _backward_euler(step, euler):	
    def dydx(x,y): return (1.+x*y)**2
    f,ax=plt.subplots(1,1)
    f.set_size_inches([12,5])
    # initialise ODE
    x=[0,]
    y=[1,]
    h0=0.08
    
    xlim=np.array([-0.05,0.75])
    ylim=[0.1,6]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)	
    
    for i in range(int(np.ceil(step))):
        ynew=root(root_equation, y[-1], args=(y[-1], h0, x[-1], dydx))
        y.append(ynew.x)
        x.append(x[-1]+h0)
    
    if not euler:
        j=abs(step-int(step))
        
        xi,yi=[x[-1], y[-1]]
        x=x[:-1]
        y=y[:-1]
        
        dy1=0.2*(yi-y[-1])
        dy2=1.5*(yi-y[-1])
        dy3=0.7*(yi-y[-1])
        dy4=(yi-y[-1])
        y1=y[-1]+dy1
        y2=y[-1]+dy2
        y3=y[-1]+dy3
        y4=y[-1]+dy4
        dydx1=dydx(xi, y1)
        dydx2=dydx(xi, y2)
        dydx3=dydx(xi, y3)
        dydx4=dydx(xi, y4)
        
        dx=0.05
        dy=0.7
        
        if step < 3.25:
            ha='left'
        elif step < 4.25:
            ha='center'
        else: 
            ha='right'
        
        if 0.2 < j < 0.4:
            ys=dydx1*(xlim - xi)+y1
            ax.plot(xlim, ys, 'b--')	
            ax.plot(xi,y1,'bo', mfc='w')
            
            ys=dydx1*(x[-1] - xi)+y1
            ax.arrow(x[-1],ys[0],0,y[-1][0]-ys[0],length_includes_head=True, head_length=0.12, head_width=0.01, color='b')
            ax.arrow(x[-1],y[-1][0],0,-y[-1][0]+ys[0],length_includes_head=True, head_length=0.12, head_width=0.01, color='b')
            
            ax.text(x[-1]-dx, y[-1]+dy, 'solving for $x^{'+'({:d})'.format(int(step)+1)+'}$: undershoot $x^{'+'({:d})'.format(int(step))+'}$', color='b', ha=ha, va='center')
            
        elif 0.4 < j < 0.6:
            ys=dydx1*(xlim - xi)+y1
            ax.plot(xlim, ys, 'b--', alpha=0.5)
            ys=dydx2*(xlim - xi)+y2
            ax.plot(xlim, ys, 'b--')
            ax.plot(xi,y2,'bo', mfc='w')
            
            ys=dydx2*(x[-1] - xi)+y2
            ax.arrow(x[-1],ys[0],0,y[-1][0]-ys[0],length_includes_head=True, head_length=0.12, head_width=0.01, color='b')
            ax.arrow(x[-1],y[-1][0],0,-y[-1][0]+ys[0],length_includes_head=True, head_length=0.12, head_width=0.01, color='b')
            
            ax.text(x[-1]-dx, y[-1]+dy, 'solving for $x^{'+'({:d})'.format(int(step)+1)+'}$: overshoot $x^{'+'({:d})'.format(int(step))+'}$', color='b', ha=ha, va='center')
            
        elif 0.6 < j < 0.8:
            ys=dydx1*(xlim - xi)+y1
            ax.plot(xlim, ys, 'b--', alpha=0.5)
            ys=dydx2*(xlim - xi)+y2
            ax.plot(xlim, ys, 'b--', alpha=0.5)
            ys=dydx3*(xlim - xi)+y3
            ax.plot(xlim, ys, 'b--')
            ax.plot(xi,y3,'bo', mfc='w')
            
            ys=dydx3*(x[-1] - xi)+y3
            ax.arrow(x[-1],ys[0],0,y[-1][0]-ys[0],length_includes_head=True, head_length=0.12, head_width=0.01, color='b')
            ax.arrow(x[-1],y[-1][0],0,-y[-1][0]+ys[0],length_includes_head=True, head_length=0.12, head_width=0.01, color='b')
            
            ax.text(x[-1]-dx, y[-1]+dy, 'solving for $x^{'+'({:d})'.format(int(step)+1)+'}$: undershoot $x^{'+'({:d})'.format(int(step))+'}$', color='b', ha=ha, va='center')
        else:	
            ys=dydx1*(xlim - xi)+y1
            ax.plot(xlim, ys, 'b--', alpha=0.5)
            ys=dydx2*(xlim - xi)+y2
            ax.plot(xlim, ys, 'b--', alpha=0.5)
            ys=dydx3*(xlim - xi)+y3
            ax.plot(xlim, ys, 'b--', alpha=0.5)
            ys=dydx4*(xlim - xi)+y4
            ax.plot(xlim, ys, 'k--')
            
            ax.plot(xi,yi,'ko', mfc='w', zorder=3)
    
            ax.text(x[-1]-dx, y[-1]+dy, 'solving for $x^{'+'({:d})'.format(int(step))+'}$: within tolerance', color='k', ha=ha, va='center')
            
    ax.plot(x,y,'ko-', mfc='k', label='Backward Euler', zorder=2)
    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x(t)$')
    
    if euler:
        # plot euler for comparison
        x0=[0,]
        y0=[1,]
        while len(x0) < len(x):
            y0.append(y0[-1]+h0*dydx(x0[-1], y0[-1]))
            x0.append(x0[-1]+h0)		
            
        ax.plot(x0,y0,'ko-', color=[0.7,0.7,0.7], mec=[0.7,0.7,0.7], zorder=1, label='Euler')
        
        ax.legend(loc=2)

    plt.show()
def backward_euler():
    steps=FloatSlider(value=7.25, min=1.25, max=9, step=0.25, description='steps')
    compare=Checkbox(False, description='compare Euler')
    io=interactive_output(_backward_euler, {'euler':compare,'step':steps})    
    return VBox([HBox([steps, compare]),io])

def f_int(x): return (x-2)*(x-5.5)*(x-7)/8+8
def _trapezium(know_gx, subints, area, NN):

    f,(ax,ax2) = plt.subplots(1,2,figsize=(15,6))


    # configure area boolean
    if area == 'None': area = 0
    elif area =='A0': area = 1
    elif area =='A1': area = 2
    elif area =='A2': area = 3
    elif area =='Atot': area = -1
    
    # plot function or data
    if know_gx:
        x = np.linspace(2,8,1001)
        
        y = f_int(x)
        ax.plot(x,y,'r-', label='known function, $f(x)$')
    else:
        xi = np.array([2, 3.5, 6.8, 8.])
        yi = np.array([7.8, 8.5, 8.1, 10.0])
        ax.plot(xi,yi,'kx',ms=5,mew=2,label='known data, $(x_i,y_i)$')
    
    # show subintervals
    if subints:
        if know_gx:
            N=3 	# number of subintervals
            xi = np.linspace(x[0],x[-1],N+1)
            yi = f_int(xi)
            ax.plot(xi,yi,'kx',ms=5,mew=2,label='eval. function, $g(x_i)$')
        ax.plot(xi,yi,'k--')
        # dashed vertical lines
        label = 'three subintervals'
        for xii,yii in zip(xi,yi):
            ax.plot([xii,xii],[0,yii],'k--',label=label)
            label=None
        # subinterval numbering
        if area == 0:
            for xi1,xi2,yi1,yi2,i in zip(xi[:-1],xi[1:],yi[:-1],yi[1:], range(len(xi))):
                ax.text(0.5*(xi1+xi2), 0.25*(yi1+yi2), '$I_'+'{:d}'.format(i+1)+'$', ha = 'center', va = 'center', size = 14)
        
        if area > 0:	
            i = area - 1		
            patches = []
            
            i1 = i
            i2 = i+2
            if i2 == len(xi):
                poly = np.array([list(xi[i1:])+[xi[-1],xi[i1]],list(yi[i1:])+[0,0]]).T
            else:
                poly = np.array([list(xi[i1:i2])+[xi[i2-1],xi[i1]],list(yi[i1:i2])+[0,0]]).T
            xint = xi[i1:i2]
            yint = yi[i1:i2]
                
            area = trapz(yint,xint)
            polygon = Polygon(poly, zorder=1)
            patches.append(polygon)
            p = PatchCollection(patches, color = 'r', alpha = 0.2)
            ax.add_collection(p)
            
            ax.text(np.mean(xint), 0.5*np.mean(yint), '$A_'+'{:d}'.format(i)+'$'+'\n$=$\n${:3.1f}$'.format(area), ha = 'center', va = 'center', size = 12)
        
        if area < 0:	
            patches = []
            area = trapz(yi,xi)
            poly = np.array([list(xi)+[xi[-1],xi[0]],list(yi)+[0,0]]).T
            polygon = Polygon(poly, zorder=1)
            patches.append(polygon)
            p = PatchCollection(patches, color = 'r', alpha = 0.2)
            ax.add_collection(p)
            
            ax.text(np.mean(xi), 0.5*np.mean(yi), '$A_{tot}'+'$'+'\n$=$\n${:3.1f}$'.format(area), ha = 'center', va = 'center', size = 12)
        
    else:
        if area < 0:
            
            patches = []
            if know_gx:
                poly = np.array([list(x)+[x[-1],x[0]],list(y)+[0,0]]).T
                area = trapz(y,x)
            else:
                poly = np.array([list(xi)+[xi[-1],xi[0]],list(yi)+[0,0]]).T
                area = trapz(yi,xi)
                
            polygon = Polygon(poly, zorder=1)
            patches.append(polygon)
            p = PatchCollection(patches, color = 'r', alpha = 0.2)
            ax.add_collection(p)
            
            ax.text(5., 4, 'Area = {:3.1f}'.format(area), ha='center', va = 'center')
        
    
    # plotting
    ax.set_xlabel('time',size = TEXTSIZE)
    ax.set_ylabel('temperature',size = TEXTSIZE)
    ax.set_xlim([0,10])
    ax.set_ylim([0, 15])
    ax.legend(loc=2, prop={'size':TEXTSIZE})
    
    # fit polynomial to data
    xi = np.array([2.5, 3.5, 4.5, 5.6, 8.6, 9.9, 13.0, 13.5])
    yi = np.array([24.7, 21.5, 21.6, 22.2, 28.2, 26.3, 41.7, 54.8])
    ak = fit_poly5(xi,yi)
    trapezium_method(ak,[xi[0], xi[-1]],NN,ax2)
    plt.show()

# evaluate polynomial with coefficients A at locations XI
def polyval(a,xi):
	"""Evaluautes polynomial with coefficients A at points XI.
	"""
	yi = 0.*xi
	for i,ai in enumerate(a):
		yi = yi + ai*xi**i
	return yi
# fit a fifth order polynomial
def fit_poly5(xi,yi):
    """Return coefficients of fifth order polynomial fitted to data XI,YI.
    """
    # construct Vandemonde matrix
    A = vandermonde(xi,5)
    
    # construct RHS vector
    b = rhs(xi,yi,5)
    
    # solve Ax=b 
    # (note: I am solving x = A^-1 b, which is not wildly efficient)
    Ainv = np.linalg.inv(A)
    ak = np.dot(Ainv, b)
    
    return ak

# integrate exactly a fifth order polynomial
def rhs(xi,yi,m):
	"""Return least-squares righthand side vector for data XI, YI and polynomial order M
	"""
	# preallocate vector
	rhs = np.zeros(m+1)
	# compute terms
	for i in range(m+1):
		rhs[i] = np.sum(xi**i*yi)
	
	return rhs
def vandermonde(xi,m):
	"""Return Vandermonde matrix for data XI and polynomial order M
	"""
	# preallocate matrix
	V = np.zeros((m+1,m+1))
	# loop over rows
	for i in range(m+1):
		# loop over columns
		for j in range(m+1):
			V[i,j] = np.sum(xi**(i+j))
	return V
def int_poly5(ak, xlim):
    akint = np.array([0.,]+[aki/(i+1) for i,aki in enumerate(ak)])
    return polyval(akint, xlim[1]) - polyval(akint, xlim[0])

# apply Trapezium method
def trapezium_method(ak,xlim,N,ax):
    """Apply Trapezium method with N subintervals to polynomial with coefficients
       AK over the interval XLIM.
    """
    # construct subintervals and function evaluations
    xin = np.linspace(xlim[0], xlim[1], N+1)
    yin = polyval(ak,xin)
    
    # compute integral
    dx = xin[1]-xin[0]
    area = dx/2*(yin[0] + 2*np.sum(yin[1:-1]) + yin[-1])
    area_true = int_poly5(ak,xlim)
    
    # plotting
        # data
    xi = np.array([2.5, 3.5, 4.5, 5.6, 8.6, 9.9, 13.0, 13.5])
    yi = np.array([24.7, 21.5, 21.6, 22.2, 28.2, 26.3, 41.7, 54.8])
    #ax.plot(xi,yi,'ko',mfc='w',mew=1.5,label='data')
        # interpolating function
    xv = np.linspace(xi[0],xi[-1],1001)
    yv = polyval(ak,xv)
    ax.plot(xv,yv,'r-',label='$f(x)$')
        # subintervals
    ax.plot(xin,yin,'k--x',mec='r',mew=1.5,label='subintervals')
    for xini,yini in zip(xin,yin): 
        ax.plot([xini,xini],[0,yini],'k--')
                    
    # plot upkeep
    ax.legend(loc=2, prop={'size': TEXTSIZE})
    ax.set_xlabel('time',size = TEXTSIZE)
    ax.set_ylabel('temperature',size = TEXTSIZE)
    str1 = '$A_{'+'{:d}'.format(N)+'}'+'={:3.1f}$'.format(area)
    str2 = '$A_{\infty}=$'+'${:3.1f}$'.format(area_true)
    str3 = '$\%\,\,err=$'+'${:3.1f}$'.format((area_true-area)/area_true*100)
    ax.annotate(str1+'\n'+str2+'\n'+str3, xy=(.05,.7), xycoords='axes fraction', ha='left', va='top', size = 12)
    
    ylim = ax.get_ylim()
    ax.set_ylim([0, ylim[-1]])


def trapezium():
    know_gx=Checkbox(False, description='$f(x)$ known')
    subints=Checkbox(False, description='show subintervals')
    area=Dropdown(
    options=['None', 'A0','A1','A2','Atot'],
    value='None',
    description='show area')
    N=IntSlider(value=1, min=1, max=10, step=1, description='# subintervals')
    io=interactive_output(_trapezium, {'know_gx':know_gx,'subints':subints,'area':area, 'NN':N})    
    return VBox([HBox([VBox([know_gx, subints, area]), N]),io])


def test():
    x1,x2,x3,x4,x5 = [1,2,3,3,2]
    dx = 1

    Idx = dx/2.*(x1+2*x3+x5)
    Idx2 = dx/4.*(x1+2*x2+2*x3+2*x4+x5)
    Isim = dx/6*(x1+4*x2+2*x3+4*x4+x5)
    Ir = (4*Idx2-Idx)/3.
    print(Idx)
    print(Idx2)
    print(Isim)
    print(Ir)
    pass

def main():
    test()

if __name__ == "__main__":
    main()