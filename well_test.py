import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact, fixed, interactive_output, HBox, Button, VBox, Output, IntSlider, Checkbox, FloatSlider, FloatLogSlider, Dropdown
from matplotlib.patches import Rectangle,Polygon
from scipy.special import expi, k0
from scipy.integrate import quad
from scipy.optimize import root
from functools import partial
from anaflow import get_lap_inv

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
# Moench-Neuman functions
def f_mn(c,x):
    return np.array([x[0]*np.tan(x[0]) - c,])
    # root derivative
def dfdx_mn(x):
    return np.array([[np.tan(x[0])+x[0]/np.cos(x[0])**2,],])
    # laplace space solution
def hd(g, b, s, p): 
    if isiterable(p): return np.array([hd(g,b,s,pi) for pi in p])
    # logic for number of terms - from WTAQ2
    NTMS = 30
    N=np.max([4,int(NTMS*2**((-np.log10(b)-2.)))])
    out = 0.
    en0 = None
    for n in range(N):
        # find root
        rhs = p/(s*b+p/g)
        f2 = partial(f_mn, rhs)
        f2 = partial(f_mn, p/(s*b))
        if en0 is None:
            if rhs < 1: en0 = np.sqrt(rhs)
            else: en0 = np.arctan(rhs)
        else:
            en0 = en + np.pi
        en = root(f2, [en0,]).x[0]
        xn = np.sqrt(b*en**2+p)
        xn = np.min([708., xn])
        
        # compute correction
        outi = 2*k0(xn)*np.sin(en)**2/(p*en*(0.5*en+0.25*np.sin(2*en)))
        out += outi
    return out

# classes
class Scene(object):
    def __init__(self):
        # hard-coded drawing parameters
        self._YL = 0.8 # land elevation
        self._DYC = 0.1 # width of confining layer
        self._WX,self._WW,self._DW,self._DW2 = [0.2, 0.02, 0.13, 0.02]  # well centre, well width, wellhead parameters
        self.akws = {'color':'b', 'head_length':0.02, 'head_width':0.02, 'length_includes_head':True}

class Well(object):
    def __init__(self, r, t, Q, T, c=1., H=3.5, itest=1, approx=False, semilog=False, analysis=False, image=False, barrier=False):
        self.r=r
        self.t=t
        self.Q=Q
        self.T=T
        self.c=c
        self.H=H
        self.s0 = 0.05
        self.S = 1.9e-3
        if itest == 5:
            # scale elastic storage for layer thickness
            self.S *= H/2.
        self.Sy = 1.9e-1
        self.itest=itest
        self.approx=approx
        self.semilog=semilog
        self.analysis=analysis
        self.image=image
        self.barrier=barrier
        if itest == 3:
            self.barrier = True
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
    def Neuman(self, r=None, t=None):
        r = self.r if r is None else r
        t = self.t if t is None else t

        if isiterable(r) and isiterable(t):
            raise TypeError("Neumann function does not support simultaneous r AND t vectorisation")
        elif isiterable(r):
            return np.array([self.h(ri, t)[0] for ri in r])
        # elif isiterable(t):
        #     return np.array([self.h(r, ti) for ti in t])
        else:
            return self.h(r, t)
    def h(self, r, t):
        if not isiterable(t): t = np.array([t,])
        # constants
        frac = 0.001
        Kr = self.T/self.H
        Kz = Kr*frac
        a1 = 1.e9
        # dimensionless quantities
        rd = r/self.H
        td = self.T*t/(r**2*self.S)
        # print(td)
        g,b,s = [a1*self.H*self.Sy/Kz, Kz*rd**2/Kr, self.S/self.Sy]
        # print(b)
        # hard-code parameters
        hd2 = get_lap_inv(partial(hd, g, b, s))
        return hd2(td)*self.Q/(4*np.pi*self.T)
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

        if self.itest == 5:   
            self.ax1.plot([self.sc._WX-self.sc._WW/2, self.sc._WX-self.sc._WW/2], [0, self.sc._YCu], 'k-')
            self.ax1.plot([self.sc._WX+self.sc._WW/2, self.sc._WX+self.sc._WW/2], [0, self.sc._YCu], 'k-')
            self.ax1.plot([self.sc._WX-self.sc._WW/2, self.sc._WX-self.sc._WW/2], [self.sc._YCu, self.sc._YL], 'k--')
            self.ax1.plot([self.sc._WX+self.sc._WW/2, self.sc._WX+self.sc._WW/2], [self.sc._YCu, self.sc._YL], 'k--')
            
            self.ax1.arrow(self.sc._WX+4*self.sc._WW, self.sc._YCu+(self.sc._YL-self.sc._YCu)/3, -3*self.sc._WW, 0, **self.sc.akws)
            self.ax1.arrow(self.sc._WX+4*self.sc._WW, self.sc._YCu+(self.sc._YL-self.sc._YCu)*2/3, -3*self.sc._WW, 0, **self.sc.akws)
            self.ax1.arrow(self.sc._WX-4*self.sc._WW, self.sc._YCu+(self.sc._YL-self.sc._YCu)/3, 3*self.sc._WW, 0, **self.sc.akws)
            self.ax1.arrow(self.sc._WX-4*self.sc._WW, self.sc._YCu+(self.sc._YL-self.sc._YCu)*2/3, 3*self.sc._WW, 0, **self.sc.akws)      
        else:
            self.ax1.plot([self.sc._WX-self.sc._WW/2, self.sc._WX-self.sc._WW/2], [0, self.sc._YCl], 'k--')
            self.ax1.plot([self.sc._WX+self.sc._WW/2, self.sc._WX+self.sc._WW/2], [0, self.sc._YCl], 'k--')
            self.ax1.plot([self.sc._WX-self.sc._WW/2, self.sc._WX-self.sc._WW/2], [self.sc._YCl, self.sc._YL], 'k-')
            self.ax1.plot([self.sc._WX+self.sc._WW/2, self.sc._WX+self.sc._WW/2], [self.sc._YCl, self.sc._YL], 'k-')
            
            if self.itest != 0:
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

        # if self.itest in [0, 3]:            
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
                (self.sc._WX-self.sc._WW,self.sc._YL)], fc = np.array([200,200,200])/255, zorder=3, ec='k', alpha=0.5)
            wellhead.xy[:,0] = 2*self.sc._WX - wellhead.xy[:,0]
            self.ax1.add_patch(wellhead)
    
            water = Polygon([
                (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-1.25*self.sc._DW2),
                (self.sc._WX+self.sc._WW+self.sc._DW2+3*Qv,self.sc._YL),
                (self.sc._WX+self.sc._WW+self.sc._DW2+0.5*Qv,self.sc._YL),
                (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-2.75*self.sc._DW2),
                (self.sc._WX+self.sc._WW+self.sc._DW2,self.sc._YL+self.sc._DW-1.25*self.sc._DW2)], fc = '#99CCFF', zorder=1, alpha=0.5)
            water.xy[:,0] = 2*self.sc._WX - water.xy[:,0]
            if self.itest == 4:
                water.xy[:,1] = 2*(self.sc._YL+self.sc._DW-2*self.sc._DW2) - water.xy[:,1]
            
            self.ax1.add_patch(water)
            
            self.sc._WX = 1.*self.sc._WXs
        if self.itest == 4:
            recharge = Rectangle((0.55,0), 0.45, self.sc._YL-self.s0, fc = '#99CCFF', zorder=2, ec=None)
            self.ax1.add_patch(recharge)
    def draw_drawdown(self):
        
        x = np.linspace(0,1.01,1001)
        if self.itest == 5: x = np.linspace(0,1.01,101)
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
        elif self.itest == 4:
            self.approx = False
            s0 = self.Theis(r=r)/10+self.s0
            if self.image:
                r1 = abs(x-0.9)*750
                s1 = -self.Theis(r=r1)/10+self.s0
                s = s0+s1 - self.s0
            else:
                s = s0
            s[np.where(x>0.55)] = self.s0
            self.approx = True
        elif self.itest == 5:
            s = self.Neuman(r=r)/10 + self.s0
        cl = 'k' if self.approx else 'b'

        inds = np.where(r>(750*self.sc._WW/2.))

        self.ax1.plot(x[inds], self.sc._YL-s[inds], cl+'--')
        if self.itest in [3,4]:
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
        elif self.itest == 4:
            self.approx = False
            ri = abs(self.sc._WX+self.r/750-0.9)*750
            s = self.Theis(t=tv)
            if self.image:
                s-=self.Theis(r=ri, t=tv)
            if (self.r/750+self.sc._WX)>0.55: 
                s*=0.
            self.approx = True
        elif self.itest == 5:      
            tv = np.logspace(-2,2,int(4/0.2)+1,10)
            it = np.argmin(abs(tv-self.t))      
            s = self.Neuman(t=tv)
            
            self.approx = False
            s0 = self.Theis(t=tv)
            fr = self.Sy/self.S
            self.S *= fr
            s1 = self.Theis(t=tv)
            self.S /= fr
            self.approx = True

        cl = 'k' if self.approx else 'b'
            
        sm,sr = [0.5*(s[0]+s[-1]), 0.5*(s[-1]-s[0])]
        tm,tr = [0.5*(tv[0]+tv[-1]), 0.5*(tv[-1]-tv[0])]
        self.ax2.set_xlim([0.9*tv[0], 1.1*tv[-1]])
        
        if self.itest == 5:
            self.ax2.set_ylim([0.0001, 2])
        else:
            self.ax2.set_ylim([0, 6])
            
        self.ax2.plot(tv[:it+1], s[:it+1], cl+'o', mfc='w', mec=cl, mew=1.5, 
            ms=7, label='data')
        
        self.ax2.set_xlabel('time [days]')
        self.ax2.set_ylabel('drawdown [m]')
        if self.semilog:
            self.ax2.set_xscale('log')
        if self.itest == 5:
            self.ax2.set_yscale('log')
            self.ax2.plot(tv, s0, 'r--', label='Theis, $S=bS_s$')
            self.ax2.plot(tv, s1, 'b--', label='Theis, $S=S_y$')
            self.ax2.legend()
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
def show_moenchneuman(**kwargs):
    w = Well(itest=5, semilog=True, approx=True, **kwargs)
    w.fig = plt.figure(figsize=(12,6))
    w.ax1 = plt.axes([0.05, 0.15, 0.55, 0.7])
    
    w.draw_scene()    
    w.draw_drawdown()
    w.ax2 = plt.axes([0.70, 0.15, 0.35, 0.7])
    w.draw_curve()
    
    plt.show()
def show_hunt2003(**kwargs):
    w = Well(itest=6, semilog=True, approx=True, **kwargs)
    w.fig = plt.figure(figsize=(12,6))
    w.ax1 = plt.axes([0.05, 0.15, 0.55, 0.7])
    
    w.draw_scene()    
    w.draw_drawdown()
    w.ax2 = plt.axes([0.70, 0.15, 0.35, 0.7])
    w.draw_curve()
    
    plt.show()
def show_theis_image(**kwargs):
    w = Well(itest=3, H=2, semilog=True, approx=True, barrier=True, **kwargs)
    w.fig = plt.figure(figsize=(12,6))
    w.ax1 = plt.axes([0.05, 0.15, 0.55, 0.7])
    
    w.draw_scene()    
    w.draw_drawdown()
    w.ax2 = plt.axes([0.70, 0.15, 0.35, 0.7])
    w.draw_curve()
    
    plt.show()
def show_theis_image2(**kwargs):
    w = Well(itest=4, H=2, semilog=True, approx=True, **kwargs)
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
def recharge_source():
    Q = FloatSlider(value=1000, description=r'$Q$ [m$^3$/day]', min = 500, max = 1500, step = 500, continuous_update = False)
    t = FloatLogSlider(value=1.0, description=r'$t$ [day]', base = 10, min=-1, max = 2, step = 0.2, continuous_update = False)
    r = FloatSlider(value=200, description=r'$r$ [m]', min = 100, max = 500, step = 100, continuous_update = False)
    T = FloatSlider(value=300, description=r'$T$ [m$^2$/day]', min = 100, max = 500, step = 100, continuous_update = False)
    image = Checkbox(value = False, description='image well')
    io = interactive_output(show_theis_image2, {'Q':Q,'t':t,'r':r,'T':T,'image':image})
    return VBox([HBox([Q,t,image]),HBox([T,r]),io])
def unconfined_aquifer():
    Q = FloatSlider(value=1000, description=r'$Q$ [m$^3$/day]', min = 500, max = 1500, step = 500, continuous_update = False)
    t = FloatLogSlider(value=1.0, description=r'$t$ [day]', base = 10, min=-1, max = 2, step = 0.2, continuous_update = False)
    r = FloatSlider(value=200, description=r'$r$ [m]', min = 100, max = 500, step = 100, continuous_update = False)
    T = FloatSlider(value=300, description=r'$T$ [m$^2$/day]', min = 100, max = 500, step = 100, continuous_update = False)
    H = FloatSlider(value=2, description=r'$b$ [m]', min=2, max = 5, step = 1.5, continuous_update = False)
    io = interactive_output(show_moenchneuman, {'Q':Q,'t':t,'r':r,'T':T,'H':H})
    return VBox([HBox([Q,t,H]),HBox([T,r]),io])
def streamflow_depletion():
    Q = FloatSlider(value=1000, description=r'$Q$ [m$^3$/day]', min = 500, max = 1500, step = 500, continuous_update = False)
    t = FloatLogSlider(value=1.0, description=r'$t$ [day]', base = 10, min=-1, max = 2, step = 0.2, continuous_update = False)
    r = FloatSlider(value=200, description=r'$r$ [m]', min = 100, max = 500, step = 100, continuous_update = False)
    T = FloatSlider(value=300, description=r'$T$ [m$^2$/day]', min = 100, max = 500, step = 100, continuous_update = False)
    H = FloatSlider(value=2, description=r'$b$ [m]', min=2, max = 5, step = 1.5, continuous_update = False)
    io = interactive_output(show_hunt2003, {'Q':Q,'t':t,'r':r,'T':T,'H':H})
    return VBox([HBox([Q,t,H]),HBox([T,r]),io])
def all_options(analysis=True):
    options = Dropdown(options = {'confined':1, 'leaky aquifer':2, 'flow barrier':3, 'recharge source':4, 'unconfined':5}, value = 1, description='Aquifer type')
    approx = Checkbox(value = True, description='approx.')
    semilog = Checkbox(value = False, description='SemiLog')
    image = Checkbox(value = False, description='image')
    Q = FloatSlider(value=1000, description=r'$Q$ [m$^3$/day]', min = 500, max = 1500, step = 500, continuous_update = False)
    t = FloatLogSlider(value=1.0, description=r'$t$ [day]', base = 10, min=-1, max = 2, step = 0.2, continuous_update = False)
    r = FloatSlider(value=200, description=r'$r$ [m]', min = 100, max = 500, step = 100, continuous_update = False)
    T = FloatSlider(value=300, description=r'$T$ [m$^2$/day]', min = 100, max = 500, step = 100, continuous_update = False)
    c = FloatLogSlider(value=1.e5, description=r'$c$ [day]', base = 10, min=2, max = 6, step = 1, continuous_update = False)
    H = FloatSlider(value=2, description=r'$b$ [m]', min=2, max = 5, step = 1.5, continuous_update = False)
    io = interactive_output(plot_theis, {'Q':Q,'t':t,'r':r,'T':T,'approx':approx,'semilog':semilog,'itest':options,'image':image,'c':c, 'H':H, 'analysis':fixed(analysis)})
    return VBox([HBox([options]),HBox([Q,t,approx]),HBox([T,r,semilog]),HBox([H,c,image]),io])

if __name__ == "__main__":
    pass
