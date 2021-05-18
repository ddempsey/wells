import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import (interact, fixed, interactive_output, 
    HBox, Button, VBox, Output, IntSlider, Checkbox, FloatSlider, 
    FloatLogSlider, Dropdown, FloatText)
from matplotlib.patches import Rectangle,Polygon
from scipy.special import expi, k0
from scipy.integrate import quad
from scipy.optimize import root
from functools import partial

ER = np.array([2.69, 4.9, 4.6])
DD = np.array([12,54,150,259,301,222,111,52,40,24,9])
ER = np.pad(ER, [0,len(DD)-len(ER)])
_COLOR = ['r','b','g','m','y','c',[0.5,0.5,1],[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,0.5],]

def check(yest, ytrue):
	return bool(abs(yest - ytrue)/abs(ytrue) < 0.5e-2)

class Hydrograph(object):
    def __init__(self, er, dd, dt=1):
        self.er = er
        self.dd = dd
        self.n = len(dd)
        self.dt = dt
        self.t = np.arange(1, self.n+1)*self.dt
        self.c = _COLOR[:self.n]
        self.solve_hydrograph()
    def solve_hydrograph(self):
        # set up matrix problem
        self.A = np.array([np.pad(self.er,[i,0])[:self.n] for i in range(self.n)]).T
        self.uh = np.linalg.solve(self.A, self.dd)        
    def get_correct(self, kwargs):
        n = 0
        for i in range(len(kwargs.values())):
            if check(kwargs['item{:02d}'.format(i)], self.uh[i]):
                n+=1
        return n
    def rainfall(self, **kwargs):
        dd = kwargs.pop('dd')
        f = plt.figure(figsize=(5,5))
        ax = plt.axes([0.1, 0.1, 0.85,0.85])
        ax.bar(self.t,self.er,self.dt, alpha=0.5, color=[1,1,1], edgecolor='k')
        for ti, eri in zip(self.t, self.er):
            ax.text(ti,eri,'{:3.2f}'.format(eri), size=8, ha='center', va='bottom')
        
        if dd == 1:
            ax.bar(self.t,self.er,self.dt,alpha=0.5,color=self.c)
        plt.show()
    def unit_hydrograph(self,**kwargs):
        dd = kwargs.pop('dd')
        f = plt.figure(figsize=(5,5))
        ax = plt.axes([0.1, 0.1, 0.85,0.85])
        vs = np.array([kwargs['item{:02d}'.format(i)] for i in range(self.n-2)])
        ax.set_ylim([0, 1.3*np.max(self.uh)])
        ax.bar(self.t[:self.n-2],vs,self.dt, alpha=0.5, color=[1,1,1], edgecolor='k')
        
        if dd == 2:
            ax.bar(self.t[:self.n-2],vs,self.dt, alpha=0.5, color=self.c[:self.n-2])

        for ti, eri in zip(self.t, vs):
            ax.text(ti,eri,'{:3.2f}'.format(eri), size=8, ha='center', va='bottom')
        plt.show()
    def streamflow(self,**kwargs):
        dd = kwargs.pop('dd')
        f = plt.figure(figsize=(5,5))
        ax = plt.axes([0.1, 0.1, 0.85,0.85])
        ax.bar(self.t,self.dd,self.dt, alpha=0.5, color=[1,1,1], edgecolor='k')
        
        n = self.get_correct(kwargs)
        
        if dd == 1:
            vs = np.array([kwargs['item{:02d}'.format(i)] for i in range(self.n-2)])
            bottom = 0.*vs
            for i in range(n):
                v = np.pad(self.er[i]*vs, [i,0])[:self.n-2]*np.sign(vs)
                ax.bar(self.t[:self.n-2], v, self.dt, alpha=0.5, 
                    color=self.c[i], bottom= bottom)
                bottom += v
        elif dd == 2:
            if n>0:
                bottom = 0.*self.t
                for i in range(n):
                    v = self.uh[i]*self.A[:,i].T
                    ax.bar(self.t, v, self.dt, alpha=0.5, color=self.c[i], bottom= bottom)
                    bottom += v
        
        if n < len(kwargs.items()):
            v = kwargs['item{:02d}'.format(n)]
            ax.bar(self.t, v*self.er, self.dt, alpha=0.25, color=self.c[n])
        for ti, eri in zip(self.t, self.dd):
            ax.text(ti,eri,'{:d}'.format(int(eri)), size=8, ha='center', va='bottom')
        plt.show()
    def linked_widgets(self):	
        items = []
        N = self.n-2
        from IPython.core.display import display, HTML
        for i in range(N):
            ft = FloatText(value = 0., description='$U_'+'{:d}'.format(i+1)+'$', disabled=True)
            # ft.add_class("right-spacing-class")
            items.append(ft)
        # display(HTML("<style>.right-spacing-class {margin-right: 200px;}</style>"))
        items[0].disabled=False
                
        def box_change(i, change):
            items[i+1].disabled = not check(change.new, self.uh[i])
            
        def box_disabled(i, change):
            if change.new: 
                for item in items[i+1:]:
                    item.disabled = True 
            else:
                for item, item0, uhi in zip(items[i+1:], items[i:], self.uh[i:]):
                    item.disabled = (not check(item0.value, uhi))

        for i in range(N-1):
            items[i].observe(partial(box_change, i), names = 'value')
            items[i].observe(partial(box_disabled, i), names = 'disabled')
                
        return dict(('item{:02d}'.format(i),item) for i,item in enumerate(items))

def isiterable(a):
    try:
        [_ for _ in a]
        return True
    except TypeError:
        return False

def hydrograph_exercise():

    h = Hydrograph(ER,DD,dt=0.5)
    dd = Dropdown(options = {'rainfall':1, 'unit hydrograph':2}, 
        value = 1, description='highlight')
    
    items = h.linked_widgets()
    its = list(items.values())
    N = int(np.ceil(len(its)/3.))
    items_list = [VBox(its[:N]), VBox(its[N:2*N]), VBox(its[2*N:]), dd]

    items.update({'dd':dd})
    io1 = interactive_output(h.rainfall, items)
    io2 = interactive_output(h.unit_hydrograph, items)
    io3 = interactive_output(h.streamflow, items)
    
    return VBox([HBox([io1,io2,io3]),HBox(items_list)])
        
if __name__ == "__main__":
    hydrograph_exercise()
