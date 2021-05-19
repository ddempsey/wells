import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import (interact, fixed, interactive_output, 
    HBox, Button, VBox, Output, IntSlider, Checkbox, FloatSlider, 
    FloatLogSlider, Dropdown, FloatText, Label, Layout)
from matplotlib.patches import Rectangle,Polygon
from scipy.special import expi, k0
from scipy.integrate import quad
from scipy.optimize import root
from functools import partial

_FS = [4,4]
ER = np.array([2.69, 4.9, 4.6])
DD = np.array([12,54,150,259,301,222,111,52,40,24,9])
ER = np.pad(ER, [0,len(DD)-len(ER)])
_COLOR = ['r','b','g','m','y','c',[0.5,0.5,1],[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,0.5],]

def check(yest, ytrue):
	return bool(abs(yest - ytrue)/abs(ytrue) < 2.e-2)

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
        hint = kwargs.pop('hint')
        f = plt.figure(figsize=_FS)
        ax = plt.axes([0.1, 0.1, 0.85,0.85])
        ax.set_xlabel('time [hr]')
        ax.set_ylabel('excess rainfall [cm]')
        ax.bar(self.t,self.er,self.dt, alpha=0.5, color=[1,1,1], edgecolor='k')
        n = self.get_correct(kwargs)
        i = np.min([int(np.sum(np.sign(self.er)))-1,n])
        for j,ti, eri in zip(range(len(self.t)), self.t, self.er):
            c = 'k'
            if dd == 1 and hint and i == j:
                c = 'r'
            ax.text(ti,eri,'{:2.1f}'.format(eri), size=8, ha='center', va='bottom', color = c)
        
        if dd == 1:
            ax.bar(self.t,self.er,self.dt,alpha=0.5,color=self.c)
        plt.show()
    def unit_hydrograph(self,**kwargs):
        dd = kwargs.pop('dd')
        hint = kwargs.pop('hint')
        f = plt.figure(figsize=_FS)
        ax = plt.axes([0.1, 0.1, 0.85,0.85])
        ax.set_xlabel('time')
        ax.set_ylabel('discharge per unit rainfall [m$^3$/s/cm]')
        vs = np.array([kwargs['item{:02d}'.format(i)] for i in range(self.n-2)])
        ax.set_ylim([0, 1.3*np.max(self.uh)])
        ax.bar(self.t[:self.n-2],vs,self.dt, alpha=0.5, color=[1,1,1], edgecolor='k')
        
        n = self.get_correct(kwargs)
        if n == 0 and hint:
            ax.text(self.t[0], vs[0], '{:d}$\div${:2.1f}\n$\downarrow$\n '.format(self.dd[0], self.er[0]), 
                ha='center', va='bottom', color='r')

        if dd == 2:
            ax.bar(self.t[:self.n-2],vs,self.dt, alpha=0.5, color=self.c[:self.n-2])

        i = np.min([int(np.sum(np.sign(self.er)))-1,n])
        for j, ti, eri in zip(range(len(self.t)), self.t, vs):
            c = 'k'
            if dd == 1 and hint and n>0 and (n-i) == j:
                c = 'r'
            ax.text(ti,eri,'{:2.1f}'.format(eri), size=8, ha='center', va='bottom', color=c)

        plt.show()
    def streamflow(self,**kwargs):
        dd = kwargs.pop('dd')
        hint = kwargs.pop('hint')
        f = plt.figure(figsize=_FS)
        ax = plt.axes([0.1, 0.1, 0.85,0.85])
        ax.bar(self.t,self.dd,self.dt, alpha=0.5, color=[1,1,1], edgecolor='k')
        ax.set_xlabel('time [hr]')
        ax.set_ylabel('discharge [m$^3$/s]')
        
        n = self.get_correct(kwargs)
        
        if dd == 1:
            vs = np.array([kwargs['item{:02d}'.format(i)] for i in range(self.n-2)])
            bottom = 0.*vs
            bin = 0
            for er,c in zip(self.er,self.c):
                i = self.c.index(c)
                v = np.pad(er*vs, [i,0])[:self.n-2]
                if n < len(v):
                    vn = v[n]
                v = v*np.sign(vs)
                if n < len(v):
                    v[n] = vn
                if hint and n < len(v): 
                    ax.bar(self.t[n], v[n], self.dt, alpha=0.25, 
                        color=self.c[i], bottom=bin)
                    bin += v[n]
                    if i == np.min([int(np.sum(np.sign(self.er)))-1,n]) and n>0:
                        ax.text(self.t[n]+self.dt/2., bin-v[n]/2., 
                            '$\leftarrow=${:2.1f}'.format(self.er[i])+r'$\times$'+'{:2.1f} '.format(vs[n-i]), 
                            ha='left', va='center', color='r')
                if n < len(v):
                    v[n] = 0.
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
        
        for ti, eri in zip(self.t, self.dd):
            ax.text(ti,eri,'{:d}'.format(int(eri)), size=8, ha='center', va='bottom')
        plt.show()
    def hintbox(self, **kwargs):
        dd = kwargs.pop('dd')
        hint = kwargs.pop('hint')
        n = self.get_correct(kwargs)

        f = plt.figure(figsize=(18,1))
        ax = plt.axes([0,0,1,1])
        ax.axis('off')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        if hint:
            if n == 0:
                text = 'Compute the ratio of rainfall to discharge in the first time step.'
            if n > 0:
                text = 'In the current discharge step, subtract the runoff due to all rainfall time steps except the first.\n'
                text += 'What remains is the delayed runoff due to rainfall in the first time step.\n'
                text += 'Compute the ratio of initial rain fall with this remainder discharge.'
            ax.text(0.5, 1, text, color='r', ha='center', va = 'top', size=18)
        plt.show()
    def linked_widgets(self):	
        items = []
        N = self.n-2
        hint = Checkbox(value=False, description='show hint')
        for i in range(N):
            ft = FloatText(value = 0., description='$U_'+'{:d}'.format(i+1)+'$', disabled=True)
            items.append(ft)
        items[0].disabled=False
                
        def box_change(i, change):
            items[i+1].disabled = not check(change.new, self.uh[i])
            if check(change.new, self.uh[i]):
                hint.value=False            
            
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

        itemsd = dict(('item{:02d}'.format(i),item) for i,item in enumerate(items))
        itemsd.update({'hint':hint}) 
        return itemsd

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
    hint = items.pop('hint')
    its = list(items.values())
    N = int(np.ceil(len(its)/3.))
    items_list = [VBox(its[:N]), VBox(its[N:2*N]), VBox(its[2*N:]), VBox([dd,hint])]

    items.update({'dd':dd,'hint':hint})
    io1 = interactive_output(h.rainfall, items)
    io2 = interactive_output(h.unit_hydrograph, items)
    io3 = interactive_output(h.streamflow, items)
    io4 = interactive_output(h.hintbox, items)
    
    return VBox([HBox([io1,io2,io3]),io4,HBox(items_list)])
        
if __name__ == "__main__":
    hydrograph_exercise()
