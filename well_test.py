import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact, fixed, interactive_output, HBox, Button, VBox, Output, IntSlider, Checkbox, FloatSlider, FloatLogSlider, Dropdown
from matplotlib.patches import Rectangle,Polygon
from scipy.special import expi
from scipy.integrate import quad
from functools import partial

def isiterable(a):
    try:
        [_ for _ in a]
        return True
    except TypeError:
        return False

# well functions
def W(u):       # Theis
    return -expi(-u)
def Wh(u, rp):  # Hantush
    """ Returns values of the Hantush well function.

        Parameters:
        -----------
        u : float, array-like
            Lower limit of Hantush integral.
        rp : float, array-like
            Aquitard hydraulic resistance term.

        Notes:
        ------
        This function is vectorized by recursion.
    """
    # check vectorisation
    if isiterable(u):
        if isiterable(rp):
            # vectorised on space, r
            return np.array([Wh(ui, rpi) for ui,rpi in zip(u, rp)])
        else:
            # vectorised on time, t
            return np.array([Wh(ui, rp) for ui in u])
    else:
        # not vectorised
        I = lambda rho, y: np.exp(-y-rho**2/(4*y))/y
        I2 = partial(I, rp)
        return quad(I2, u, +np.inf)[0]

# classes
class Scene(object):
    def __init__(self):
        # hard-coded drawing parameters
        self._YL = 0.8 # land elevation
        self._DYC = 0.1 # width of confining layer
        self._WX,self._WW,self._DW,self._DW2 = [0.2, 0.02, 0.13, 0.02]  # well centre, well width, wellhead parameters
        self.akws = {'color':'b', 'head_length':0.02, 'head_width':0.02, 'length_includes_head':True}

class Well(object):
    def __init__(self, r, t, Q, T, c=1., H=3.5, itest=1, approx=False, semilog=False, analysis=False, image=False, barrier=True):
        self.r=r
        self.t=t
        self.Q=Q
        self.T=T
        self.c=c
        self.H=H
        self.s0 = 0.05
        self.S = 1.9e-3
        self.Sy = 1.9e-1
        self.itest=itest
        self.approx=approx
        self.semilog=semilog
        self.analysis=analysis
        self.image=image
        self.barrier=barrier
        self.ax1 = None
        self.ax2 = None
        self.sc = Scene()
    def __repr__(self):
        return self.test
    # type curves
    def Theis(self, r=None, t=None):
        r = self.r if r is None else r
        t = self.t if t is None else t
        if self.approx:
            return 2.3*self.Q/(4*np.pi*self.T)*np.log10(2.25*self.T*t/(r**2*self.S))
        else:
            return self.Q/(4*np.pi*self.T)*W(r**2*self.S/(4*self.T*t))
    def HantushJacob(self, r=None, t=None):
        r = self.r if r is None else r
        t = self.t if t is None else t
        return self.Q/(4*np.pi*self.T)*Wh(r**2*self.S/(4*self.T*t), r/np.sqrt(self.T*self.c))
    # plotting functions
    def draw_scene(self, labels=False):
        self.sc._YCu = self.sc._YL-self.H/10
        self.sc._YCl = self.sc._YCu - self.sc._DYC
        self.ax1.axhline(self.sc._YL,color='k',linestyle='-')
        self.ax1.set_xlim([0,1])
        self.ax1.set_ylim([0,1])
        
        sky = Rectangle((0,self.sc._YL), 1, 1-self.sc._YL, fc = 'b', zorder=0, alpha=0.1)
        self.ax1.add_patch(sky)
        
        ground = Rectangle((0,0), 1, self.sc._YL, fc = np.array([209,179,127])/255, zorder=0, alpha=0.7)
        self.ax1.add_patch(ground)
        
        confining = Rectangle((0,self.sc._YCl), 1, self.sc._DYC, fc = np.array([100,100,100])/255, zorder=0, alpha=0.7)
        self.ax1.add_patch(confining)
        if labels:
            self.ax1.text(0.98, (self.sc._YCl+self.sc._YCu)/2., 'confining layer', ha='right', va='center', size=12)
            self.ax1.text(0.98, (self.sc._YCl+self.sc._YCu)/2.-self.sc._DYC, 'confined aquifer', ha='right', va='center', size=12)
            self.ax1.text(0.98, (self.sc._YCl+self.sc._YCu)/2.+self.sc._DYC, 'unconfined aquifer', ha='right', va='center', size=12)
        
        well = Rectangle((self.sc._WX-self.sc._WW/2,0), self.sc._WW, self.sc._YL, fc = np.array([200,200,200])/255, zorder=1, ec=None)
        self.ax1.add_patch(well)
                    
        # else:
        self.ax1.plot([self.sc._WX-self.sc._WW/2, self.sc._WX-self.sc._WW/2], [0, self.sc._YCl], 'k--')
        self.ax1.plot([self.sc._WX+self.sc._WW/2, self.sc._WX+self.sc._WW/2], [0, self.sc._YCl], 'k--')
        self.ax1.plot([self.sc._WX-self.sc._WW/2, self.sc._WX-self.sc._WW/2], [self.sc._YCl, self.sc._YL], 'k-')
        self.ax1.plot([self.sc._WX+self.sc._WW/2, self.sc._WX+self.sc._WW/2], [self.sc._YCl, self.sc._YL], 'k-')
        
        self.ax1.arrow(self.sc._WX+4*self.sc._WW, self.sc._YCl/3, -3*self.sc._WW, 0, **self.sc.akws)
        self.ax1.arrow(self.sc._WX+4*self.sc._WW, self.sc._YCl*2/3, -3*self.sc._WW, 0, **self.sc.akws)
        self.ax1.arrow(self.sc._WX-4*self.sc._WW, self.sc._YCl/3, 3*self.sc._WW, 0, **self.sc.akws)
        self.ax1.arrow(self.sc._WX-4*self.sc._WW, self.sc._YCl*2/3, 3*self.sc._WW, 0, **self.sc.akws)

        well2 = Rectangle((self.sc._WX+self.r/750-self.sc._WW/4,0), self.sc._WW/2, self.sc._YL, fc = np.array([200,200,200])/255, zorder=1, ec='k')
        self.ax1.add_patch(well2)
        if labels:
            self.ax1.text(self.sc._WX+self.r/750, 
                self.sc._YL+self.sc._DYC/1.5, 'observation\nwell', ha='center', va='center', size=12)
            self.ax1.text(0.02, self.sc._YL-self.s0-self.sc._DYC/4, 'piezometric\nsurface', ha='left', va='top', size=12)
        
        wellhead = Polygon([
            (self.sc._WX-self.sc._WW,self.sc._YL),
            (self.sc._WX-self.sc._WW,self.sc._YL+self.sc._DW),
            (self.sc._WX+self.sc._WW,self.sc._YL+self.sc._DW),
            (self.sc._WX+self.sc._WW,self.sc._YL+self.sc._DW-self.sc._DW2),
            (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-self.sc._DW2),
            (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-3*self.sc._DW2),
            (self.sc._WX+self.sc._WW,self.sc._YL+self.sc._DW-3*self.sc._DW2),
            (self.sc._WX+self.sc._WW,self.sc._YL),
            (self.sc._WX-self.sc._WW,self.sc._YL)], fc = np.array([200,200,200])/255, zorder=2, ec='k')
        self.ax1.add_patch(wellhead)
        if labels:
            self.ax1.text(self.sc._WX+self.sc._WW+self.sc._DW2*2, 
                self.sc._YL+self.sc._DW-2*self.sc._DW2, 'pumping\nwell', ha='left', va='center', size=12)
        
        
        Qv = 0.05*self.Q/1000
        water = Polygon([
            (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-1.25*self.sc._DW2),
            (self.sc._WX+self.sc._WW+self.sc._DW2+3*Qv,self.sc._YL),
            (self.sc._WX+self.sc._WW+self.sc._DW2+0.5*Qv,self.sc._YL),
            (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-2.75*self.sc._DW2),
            (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-1.25*self.sc._DW2)], fc = '#99CCFF', zorder=1)
        self.ax1.add_patch(water)
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])

        if self.itest in [0, 3]:            
            if self.barrier:
                barrier = Rectangle((0.55-self.sc._WW/2,0), self.sc._WW, self.sc._YCl, fc = np.array([100,100,100])/255, zorder=1, ec=None, alpha=0.7)
                self.ax1.add_patch(barrier)
                if labels:
                    self.ax1.text(0.55+self.sc._WW, 0.02, 'flow barrier', ha='left', va='bottom', size=12)


            if self.image:
                # draw mirror well
                self.sc._WXs = 1.*self.sc._WX
                self.sc._WX = 0.9
                well = Rectangle((self.sc._WX-self.sc._WW/2,0), self.sc._WW, self.sc._YL, fc = np.array([200,200,200])/255, zorder=1, ec=None, alpha=0.5)
                self.ax1.add_patch(well)
                            
                # else:
                self.ax1.plot([self.sc._WX-self.sc._WW/2, self.sc._WX-self.sc._WW/2], [0, self.sc._YCl], 'k--', alpha=0.5)
                self.ax1.plot([self.sc._WX+self.sc._WW/2, self.sc._WX+self.sc._WW/2], [0, self.sc._YCl], 'k--', alpha=0.5)
                self.ax1.plot([self.sc._WX-self.sc._WW/2, self.sc._WX-self.sc._WW/2], [self.sc._YCl, self.sc._YL], 'k-', alpha=0.5)
                self.ax1.plot([self.sc._WX+self.sc._WW/2, self.sc._WX+self.sc._WW/2], [self.sc._YCl, self.sc._YL], 'k-', alpha=0.5)
                
                self.ax1.arrow(self.sc._WX+4*self.sc._WW, self.sc._YCl/3, -3*self.sc._WW, 0, alpha=0.5, **self.sc.akws)
                self.ax1.arrow(self.sc._WX+4*self.sc._WW, self.sc._YCl*2/3, -3*self.sc._WW, 0, alpha=0.5, **self.sc.akws)
                self.ax1.arrow(self.sc._WX-4*self.sc._WW, self.sc._YCl/3, 3*self.sc._WW, 0, alpha=0.5, **self.sc.akws)
                self.ax1.arrow(self.sc._WX-4*self.sc._WW, self.sc._YCl*2/3, 3*self.sc._WW, 0, alpha=0.5, **self.sc.akws)

                wellhead = Polygon([
                    (self.sc._WX-self.sc._WW,self.sc._YL),
                    (self.sc._WX-self.sc._WW,self.sc._YL+self.sc._DW),
                    (self.sc._WX+self.sc._WW,self.sc._YL+self.sc._DW),
                    (self.sc._WX+self.sc._WW,self.sc._YL+self.sc._DW-self.sc._DW2),
                    (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-self.sc._DW2),
                    (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-3*self.sc._DW2),
                    (self.sc._WX+self.sc._WW,self.sc._YL+self.sc._DW-3*self.sc._DW2),
                    (self.sc._WX+self.sc._WW,self.sc._YL),
                    (self.sc._WX-self.sc._WW,self.sc._YL)], fc = np.array([200,200,200])/255, zorder=2, ec='k', alpha=0.5)
                wellhead.xy[:,0] = 2*self.sc._WX - wellhead.xy[:,0]
                self.ax1.add_patch(wellhead)
        
                water = Polygon([
                    (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-1.25*self.sc._DW2),
                    (self.sc._WX+self.sc._WW+self.sc._DW2+3*Qv,self.sc._YL),
                    (self.sc._WX+self.sc._WW+self.sc._DW2+0.5*Qv,self.sc._YL),
                    (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-2.75*self.sc._DW2),
                    (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-1.25*self.sc._DW2)], fc = '#99CCFF', zorder=1, alpha=0.5)
                water.xy[:,0] = 2*self.sc._WX - water.xy[:,0]
                self.ax1.add_patch(water)
                
                self.sc._WX = 1.*self.sc._WXs
    def draw_drawdown(self):
        
        x = np.linspace(0,1.01,1001)
        r = abs(x-self.sc._WX)*750
        if self.itest == 0:
            s = 0.*r + self.s0
        elif self.itest == 1:
            s = self.Theis(r=r)/10 + self.s0
        elif self.itest == 2:
            s = self.HantushJacob(r=r)/10 + self.s0
        elif self.itest == 3:
            self.approx = False
            s0 = self.Theis(r=r)/10+self.s0
            if self.image:
                r1 = abs(x-0.9)*750
                s1 = self.Theis(r=r1)/10+self.s0
                s = s0+s1 - self.s0
            else:
                s = s0
            s[np.where(x>0.55)] = self.s0
            self.approx = True
        cl = 'k' if self.approx else 'b'

        inds = np.where(r>(750*self.sc._WW/2.))

        self.ax1.plot(x[inds], self.sc._YL-s[inds], cl+'--')
        if self.itest == 3:
            self.ax1.plot(x[inds], self.sc._YL-s0[inds], cl+'--', alpha=0.5)
            if self.image:
                inds = np.where(r1>(750*self.sc._WW/2.))
                self.ax1.plot(x[inds], self.sc._YL-s1[inds], cl+'--', alpha=0.5)
        self.ax1.fill_between(x[inds], 0.*s[inds], self.sc._YL-s[inds], color = 'k', alpha=0.1)

        i = np.argmin(abs(x-(self.sc._WX+self.r/750)))
        well2 = Rectangle((x[i]-self.sc._WW/4,0), self.sc._WW/2, (self.sc._YL-s[i]), fc = '#99CCFF', zorder=1, ec='k')
        self.ax1.add_patch(well2)
        
        if self.itest == 2:
            for ri in [0.1, 0.3, 0.5, 0.7,0.9]:
                i = np.argmin(abs(x-ri))
                self.ax1.arrow(ri, 0.75*self.sc._YCl+0.25*self.sc._YCu, 0, -s[i]*0.8-self.sc._DYC/4., **self.sc.akws)
    def draw_curve(self):
        tv = np.logspace(-1,2,int(3/0.2)+1,10)
        it = np.argmin(abs(tv-self.t))
        
        if self.itest == 1:
            s = self.Theis(t=tv)
        elif self.itest == 2:
            s = self.HantushJacob(t=tv)
        elif self.itest == 3:
            self.approx = False
            ri = abs(self.sc._WX+self.r/750-0.9)*750
            s = self.Theis(t=tv)
            if self.image:
                s+=self.Theis(r=ri, t=tv)
            if (self.r/750+self.sc._WX)>0.55: 
                s*=0.
            self.approx = True
        cl = 'k' if self.approx else 'b'
            
        sm,sr = [0.5*(s[0]+s[-1]), 0.5*(s[-1]-s[0])]
        tm,tr = [0.5*(tv[0]+tv[-1]), 0.5*(tv[-1]-tv[0])]
        self.ax2.set_xlim([0.9*tv[0], 1.1*tv[-1]])
        self.ax2.set_ylim([0, 6])
        s = s[:it+1]
        tv = tv[:it+1]
        self.ax2.plot(tv, s, cl+'o', mfc='w', mec=cl, mew=1.5, ms=7)
        
        self.ax2.set_xlabel('time [days]')
        self.ax2.set_ylabel('drawdown [m]')
        if self.semilog:
            self.ax2.set_xscale('log')
        self.ax2.xaxis.grid(which='minor')
        self.ax2.yaxis.grid()
    
# drawing functions
def show_aquifer(r, barrier):
    w = Well(r=r, itest=0, Q=1, t=1, T=1, H=2, approx=True, barrier=barrier)
    w.fig = plt.figure(figsize=(12,6))
    w.ax1 = plt.axes([0.05, 0.15, 0.55, 0.7])
    w.draw_scene(labels=True)  
    w.draw_drawdown()  
    plt.show()
def show_theis(**kwargs):
    w = Well(itest=1, H=2, **kwargs)
    w.fig = plt.figure(figsize=(12,6))
    w.ax1 = plt.axes([0.05, 0.15, 0.55, 0.7])
    
    w.draw_scene()    
    w.draw_drawdown()

    if w.analysis:
        w.ax2 = plt.axes([0.70, 0.15, 0.35, 0.7])
        w.draw_curve()
    
    plt.show()
def show_hantushjacob(**kwargs):
    w = Well(itest=2, H=2, semilog=True, approx=True, **kwargs)
    w.fig = plt.figure(figsize=(12,6))
    w.ax1 = plt.axes([0.05, 0.15, 0.55, 0.7])
    
    w.draw_scene()    
    w.draw_drawdown()
    w.ax2 = plt.axes([0.70, 0.15, 0.35, 0.7])
    w.draw_curve()
    
    plt.show()
def show_theis_image(**kwargs):
    w = Well(itest=3, H=2, semilog=True, approx=True, **kwargs)
    w.fig = plt.figure(figsize=(12,6))
    w.ax1 = plt.axes([0.05, 0.15, 0.55, 0.7])
    
    w.draw_scene()    
    w.draw_drawdown()
    w.ax2 = plt.axes([0.70, 0.15, 0.35, 0.7])
    w.draw_curve()
    
    plt.show()
def plot_theis(**kwargs):
    w = Well(**kwargs)
    w.fig = plt.figure(figsize=(12,6))
    w.ax1 = plt.axes([0.05, 0.15, 0.55, 0.7])
    
    w.draw_scene()    
    w.draw_drawdown()

    if w.analysis:
        w.ax2 = plt.axes([0.70, 0.15, 0.35, 0.7])
        w.draw_curve()
    
    plt.show()

# widget functions
def conceptual_model():
    barrier = Checkbox(value = False, description='Flow barrier')
    r = FloatSlider(value=200, description=r'$r$ [m]', min = 100, max = 500, step = 100, continuous_update = False)
    io = interactive_output(show_aquifer, {'r':r, 'barrier':barrier})
    return VBox([HBox([r, barrier]),io])
def confined_aquifer(analysis=False):
    approx = Checkbox(value = True, description='approx.')
    semilog = Checkbox(value = False, description='SemiLog')
    Q = FloatSlider(value=1000, description=r'$Q$ [m$^3$/day]', min = 500, max = 1500, step = 500, continuous_update = False)
    t = FloatLogSlider(value=1.0, description=r'$t$ [day]', base = 10, min=-1, max = 2, step = 0.2, continuous_update = False)
    r = FloatSlider(value=200, description=r'$r$ [m]', min = 100, max = 500, step = 100, continuous_update = False)
    T = FloatSlider(value=300, description=r'$T$ [m$^2$/day]', min = 100, max = 500, step = 100, continuous_update = False)
    io = interactive_output(show_theis, {'Q':Q,'t':t,'r':r,'T':T,'approx':approx,'semilog':semilog,'analysis':fixed(analysis)})
    return VBox([HBox([Q,t,approx]),HBox([T,r,semilog]),io])
def leaky_aquifer():
    Q = FloatSlider(value=1000, description=r'$Q$ [m$^3$/day]', min = 500, max = 1500, step = 500, continuous_update = False)
    t = FloatLogSlider(value=1.0, description=r'$t$ [day]', base = 10, min=-1, max = 2, step = 0.2, continuous_update = False)
    r = FloatSlider(value=200, description=r'$r$ [m]', min = 100, max = 500, step = 100, continuous_update = False)
    T = FloatSlider(value=300, description=r'$T$ [m$^2$/day]', min = 100, max = 500, step = 100, continuous_update = False)
    c = FloatLogSlider(value=1.e5, description=r'$c$ [day]', base = 10, min=2, max = 6, step = 1, continuous_update = False)
    io = interactive_output(show_hantushjacob, {'Q':Q,'t':t,'r':r,'T':T,'c':c})
    return VBox([HBox([Q,t,c]),HBox([T,r]),io])
def flow_barrier():
    Q = FloatSlider(value=1000, description=r'$Q$ [m$^3$/day]', min = 500, max = 1500, step = 500, continuous_update = False)
    t = FloatLogSlider(value=1.0, description=r'$t$ [day]', base = 10, min=-1, max = 2, step = 0.2, continuous_update = False)
    r = FloatSlider(value=200, description=r'$r$ [m]', min = 100, max = 500, step = 100, continuous_update = False)
    T = FloatSlider(value=300, description=r'$T$ [m$^2$/day]', min = 100, max = 500, step = 100, continuous_update = False)
    image = Checkbox(value = True, description='image well')
    io = interactive_output(show_theis_image, {'Q':Q,'t':t,'r':r,'T':T,'image':image})
    return VBox([HBox([Q,t,image]),HBox([T,r]),io])
def all_options(analysis=False):
    options = Dropdown(options = {'confined':1, 'leaky aquifer':2, 'flow barrier':3}, value = 2, description='Aquifer type')
    approx = Checkbox(value = True, description='approx.')
    semilog = Checkbox(value = False, description='SemiLog')
    Q = FloatSlider(value=1000, description=r'$Q$ [m$^3$/day]', min = 500, max = 1500, step = 500, continuous_update = False)
    t = FloatLogSlider(value=1.0, description=r'$t$ [day]', base = 10, min=-1, max = 2, step = 0.2, continuous_update = False)
    r = FloatSlider(value=200, description=r'$r$ [m]', min = 100, max = 500, step = 100, continuous_update = False)
    T = FloatSlider(value=300, description=r'$T$ [m$^2$/day]', min = 100, max = 500, step = 100, continuous_update = False)
    c = FloatLogSlider(value=1.e5, description=r'$c$ [day]', base = 10, min=2, max = 6, step = 1, continuous_update = False)
    H = FloatSlider(value=2, description=r'$H$ [m]', min=2, max = 5, step = 1.5, continuous_update = False)
    io = interactive_output(plot_theis, {'Q':Q,'t':t,'r':r,'T':T,'approx':approx,'semilog':semilog,'itest':options,'c':c, 'H':H, 'analysis':fixed(analysis)})
    return VBox([HBox([options]),HBox([Q,t,approx]),HBox([T,r,semilog]),HBox([H,c]),io])

if __name__ == "__main__":
    pass