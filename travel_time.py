import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.special import lambertw as LW
from scipy.optimize import root
from scipy.integrate import solve_ivp

from ipyleaflet import Map, basemaps, Marker, Polyline, Polygon, DivIcon, AwesomeIcon
from ipywidgets import HBox, VBox, IntSlider, FloatSlider, Button, Layout, BoundedFloatText, FloatLogSlider
from pyproj import Proj, transform

from functools import partial
from random import random

from scipy.special import expi

inProj = Proj('epsg:32759')
outProj = Proj('epsg:4326')

# class to handle widgets and updating map with contours
class GroundwaterMap(Map):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wells=[]
        self.contour_labels=[]
        self.flag=False
    def add_well(self, location, icon=None):
        well = Marker(location=location, draggable=True, icon=icon)    
        self.add_layer(well)
        self.wells.append(well)
    def update_contours(self, ps, ts):
        for p,p0 in zip(ps,self.ps0):
            p0.locations = p.locations

        [self.remove_layer(tk) for tk in self.contour_labels]

        self.contour_labels=[]
        for ti in ts:
            i = DivIcon(html=ti._text+'&nbsp;m')
            tsm = Marker(location=[ti._x, ti._y], icon=i)
            self.add_layer(tsm)
            self.contour_labels.append(tsm)
    def on_change(self,event):
        self.func()
    def configure(self, widgets, func):
        self.widgets=widgets
        self.func=func

        # initial drawdown contours        
        self.func()

        # configure interactions and feedbacks
        for w in list(self.widgets.values())+self.wells:
            w.observe(self.on_change)
        
        def on_zoom_change(self, *args, **kwargs):
            mi = args[0]['owner']
            bnds=[[mi.south, mi.west],[mi.north, mi.east]]
            func(bnds=bnds)

        self.observe(partial(on_zoom_change, self), 'zoom')
        
        def handle_map_interaction(self, type, **kwargs):
            if type!='mouseup': 
                return
            self.func()

        self.on_interaction(partial(handle_map_interaction, self))
    def travel_time(self, dhdx, n, b, xw=0, yw=0, bnds=None):
        '''
            t : float
                time [days]
            dhdx : float
                hydraulic gradient [m/m]
            Q : float
                extraction rate [m^3/s]
            n : float
                porosity [-]
            b : float
                aquifer thickness [m]
            theta : float
                angle between x axis and dominant flow direction
            xw : float
                x location of well
            yw : float
                y location of well
        '''
        
        T = 0.05
        # get parameters
        Q = self.widgets['Q'].value/1.e3
        t = self.widgets['t'].value
        theta = (270-self.widgets['th'].value)/180.*3.1416
        dhdx = self.widgets['q'].value/1.e3
        q = dhdx*T/b

        # compute contour
        if Q < 1.e-5:
            # special case, zero pumping
            xout=np.array([xw])
            yout=np.array([yw])
        else:
            # compute dimensionless solution
            t*=24*3600    # convert days to seconds
            t0 = 2.*np.pi*q**2/(b*n*Q)*t
            x0,y0 = travel_time_dimensionless(t0)
            x = Q*x0/(2*np.pi*q)
            y = Q*y0/(2*np.pi*q)
            x,y = np.concatenate([x, x[::-1]]), np.concatenate([y, -y[::-1]])
            x,y = (np.cos(theta)*x-np.sin(theta)*y, np.sin(theta)*x+np.cos(theta)*y)
            xout=x+xw
            yout=y+yw

        # add or update travel time contour
        lat,lon = xy2ll(xout,yout,*self.wells[0].location)
        try:
            self.tt_line.locations=list(zip(lat,lon))
        except AttributeError:
            self.tt_line = Polyline(locations=list(zip(lat,lon)), color='red', fill=False, weight=2)
            self.add_layer(self.tt_line)

        # add or update piezometric surface
        if bnds is None:
            bnds = self.bounds
            if len(bnds) == 0:
                bnds = ((-43.53118921794094, 172.62774467468262), 
                    (-43.506293197337435, 172.70936965942383))
        
        ps, ts = self.TheisContours(T, [Q], bnds, [dhdx, theta], levels=np.arange(-10,20,1))

        try:
            self.update_contours(ps,ts)
        except AttributeError:
            self.ps0=ps
            for polygons in self.ps0:
                self.add_layer(polygons)
    def superposition(self, T, bnds=None):
        qs = [self.widgets['Q{:d}'.format(i)].value/1.e3 for i,w in enumerate(self.wells)]

        # add or update piezometric surface
        if bnds is None:
            bnds = self.bounds
            if len(bnds) == 0:
                bnds = ((-43.54369559037467, 172.58697509765628), 
                    (-43.493903600645126, 172.7502250671387))
        ps, ts = self.TheisContours(T, qs, bnds)

        try:
            self.update_contours(ps,ts)
        except AttributeError:
            self.ps0=ps
            for polygons in self.ps0:
                self.add_layer(polygons)        
    def TheisContours(self, T, qs, bnds=None, grad=[0,0], levels=(0.5,0.75,1,1.25,1.5,1.75,2.0)):
        
            
        lats = []
        lons = []
        for w in self.wells:
            lat,lon = w.location
            lats.append(lat); lons.append(lon)
        xs,ys = transform(outProj, inProj, lats, lons)
        if bnds is None:
            xs2,ys2 = xs,ys
        else:
            for bnd in bnds:
                lat,lon = bnd
                lats.append(lat); lons.append(lon)
                xs2,ys2 = transform(outProj, inProj, lats, lons)
        n = 100
        x0,x1 = np.min(xs2), np.max(xs2)
        xr = x1-x0
        y0,y1 = np.min(ys2), np.max(ys2)
        yr = y1-y0
        xx,yy = np.meshgrid(np.linspace(x0-0.05*xr,x1+0.05*xr,n), 
            np.linspace(y0-0.05*yr,y1+0.05*yr,n))
        hh = 0.*xx+grad[0]*(np.cos(grad[1])*(xx-xs[0]) + np.sin(grad[1])*(yy-ys[0]))#/1.e3
        try:
            t = self.widgets['t'].value
        except KeyError:
            t = 100.
        for w,x,y,q in zip(self.wells,xs,ys,qs):
            if self.flag:
                hh -= Theis(np.sqrt((xx.flatten()-x)**2+(yy.flatten()-y)**2), t*24*3600, T, 1.e-4, q).reshape(xx.shape)
            else:
                hh += Theis(np.sqrt((xx.flatten()-x)**2+(yy.flatten()-y)**2), t*24*3600, T, 1.e-4, q).reshape(xx.shape)
        lat,lon=transform(inProj,outProj,xx.flatten(),yy.flatten())
        cs = plt.contourf(lat.reshape(xx.shape), lon.reshape(yy.shape), hh, 
            levels=levels, extend='both')
        ts = plt.clabel(cs, levels=[l for l,a in zip(cs.levels, cs.allsegs) if len(a)>0])
        plt.close()
        
        allsegs = cs.allsegs
        allkinds = cs.allkinds
        cmap = cm.Blues
        colors = ['#%02x%02x%02x' % tuple(int(j*255) for j in cmap(i)[:3]) for i in np.linspace(0,1,len(allsegs))]
        alphas = np.linspace(0.2,0.7,len(allsegs))
        ps = []
        for clev in range(len(cs.allsegs)):
            kinds = None if allkinds is None else allkinds[clev]
            segs = split_contours(allsegs[clev], kinds)
            polygons = Polygon(
                            locations=[p.tolist() for p in segs],
                            color='yellow',
                            weight=1,
                            opacity=1.,
                            fill_color=colors[clev],
                            fill_opacity=alphas[clev]
            )
            ps.append(polygons)
        return ps,ts

# drawdown functions
def W(u): 
    return -expi(-u)
def Theis(r,t,T,S,Q):
    return Q/(4*np.pi*T)*W(r**2*S/(4*T*t))

# travel time functions
def dydxf(x,y,t0): 
    '''
    sin(y) x/y+cos(y) = -sin(y) dy/dx + sin(y)/y - sin(y) x/y^2 dy/dx +cos(y)*x/y dy/dx

    sin(y)*x/y+cos(y)-sin(y)/y = (cos(y)*x/y-sin(y)-sin(y)*x/y^2)*dy/dx 
    
    x/y+1/tan(y)-1/y = (x/y/tan(y)-1-x/y^2)*dy/dx 
    
    x+y/tan(y)-1 = (x/tan(y)-y-x/y)*dy/dx 
    
    b = a*dy/dx
    '''
    # a = x/np.tan(y)-y-x/y
    # b = (x+y/np.tan(y)-1)
    # return np.array([np.min([(b/a)[0], 1.e20]),])
    a = np.cos(y)*x/y-np.sin(y)-np.sin(y)*x/y**2
    b = np.exp(x-t0)-np.sin(y)/y
    return np.array([np.min([(b/a)[0], 1.e20]),])
def d2ydx2(x,y,t0):
    '''
    dy/dx = b/a

    d2y/dx2 = (db/dx*a - da/dx*b)/a^2
    '''
    a = np.cos(y)*x/y-np.sin(y)-np.sin(y)*x/y**2
    b = np.exp(x-t0)-np.sin(y)/y
    dydx = dydxf(x,y,t0)
    dadx = np.cos(y)/y + x*(y*np.sin(y)+np.cos(y))/y**2*dydx-np.cos(y)*dydx-np.sin(y)/y**2-(y*np.cos(y)-2*np.sin(y))/y**3*dydx
    dbdx = np.exp(x-t0)-(y*np.cos(y)-np.sin(y))/y**2*dydx
    return np.array([np.min([((a*dbdx-b*dadx)/a**2)[0], 1.e20]),])
def travel_time_dimensionless(t0):
    ''' implementing solution from Bear and Jacobs (1965)
    '''
    # solve intercepts
    x1 = np.real(-LW(-np.exp(-t0-1),-1)-1)
    x0 = np.real(-LW(-np.exp(-t0-1))-1)
    
    # solve minor axis approx
    nx = 1000
    dx = 1.e-4*(x1-x0)
    y0 = root(lambda y, f, x, h: y-f(x,y,t0)*h, dx, args=(dydxf, x0+dx, dx)).x
    sol = solve_ivp(dydxf, [x0+dx,x1-dx], y0, method='LSODA', t_eval=np.linspace(x0+dx,x1-dx,nx), args=(t0,), jac=d2ydx2)
    sol.y[0][-1]=0 
    return sol.t, sol.y[0]

# map functions
def xy2ll(x,y,lat0,lon0):   
    x0,y0=transform(outProj, inProj, lat0, lon0)#, lat0)
    lat,lon = transform(inProj,outProj,x+x0,y+y0)
    return list(lat), list(lon)
def split_contours(segs, kinds=None):
    """takes a list of polygons and vertex kinds and separates disconnected vertices into separate lists.
    The input arrays can be derived from the allsegs and allkinds atributes of the result of a matplotlib
    contour or contourf call. They correspond to the contours of one contour level.
    
    Example:
    cs = plt.contourf(x, y, z)
    allsegs = cs.allsegs
    allkinds = cs.allkinds
    for i, segs in enumerate(allsegs):
        kinds = None if allkinds is None else allkinds[i]
        new_segs = split_contours(segs, kinds)
        # do something with new_segs
        
    More information:
    https://matplotlib.org/3.3.3/_modules/matplotlib/contour.html#ClabelText
    https://matplotlib.org/3.1.0/api/path_api.html#matplotlib.path.Path

    Solution from here:
    https://stackoverflow.com/questions/65634602/plotting-contours-with-ipyleaflet
    """
    if kinds is None:
        return segs    # nothing to be done
    # search for kind=79 as this marks the end of one polygon segment
    # Notes: 
    # 1. we ignore the different polygon styles of matplotlib Path here and only
    # look for polygon segments.
    # 2. the Path documentation recommends to use iter_segments instead of direct
    # access to vertices and node types. However, since the ipyleaflet Polygon expects
    # a complete polygon and not individual segments, this cannot be used here
    # (it may be helpful to clean polygons before passing them into ipyleaflet's Polygon,
    # but so far I don't see a necessity to do so)
    new_segs = []
    for i, seg in enumerate(segs):
        segkinds = kinds[i]
        boundaries = [0] + list(np.nonzero(segkinds == 79)[0])
        for b in range(len(boundaries)-1):
            new_segs.append(seg[boundaries[b]+(1 if b>0 else 0):boundaries[b+1]])
    return new_segs

# widget functions
def travel_time_fun():
    center=[-43.51876443245584, 172.66858981519297]
    m = GroundwaterMap(basemap=basemaps.Esri.WorldImagery, center=center, zoom=14)
    t = IntSlider(value=100, description=r'$t_t$ [day]', min=10, max = 365, step = 20, 
        continuous_update = False, layout=Layout(max_width='250px'))
    Q = FloatSlider(value=100, description=r'pumping [L/s]', min = 0, max = 200, step = 20, 
        continuous_update = False, layout=Layout(max_width='250px'))
    q = FloatSlider(value=1.5, description=r'$dh/dx$ [m/km]', min = 0.5, max = 3, step = 0.5, 
        continuous_update = False, layout=Layout(max_width='270px'))
    th = BoundedFloatText(value=135, min=0, max=180, description='flow dir. [$^{\circ}$]',layout=Layout(max_width='150px'))
    m.add_well(center)
    m.flag=True
    m.configure(widgets={'t':t, 'Q':Q, 'q':q, 'th':th}, func=partial(m.travel_time, 1.e-4, 0.03, 10.))
    return VBox([m, HBox([Q, t, q, th])])
def superposition_fun(T=0.025):
    center=[-43.51876443245584, 172.66858981519297]
    m = GroundwaterMap(basemap=basemaps.Esri.WorldImagery, center=center, zoom=13)
    Qs = []
    bs = []
    for c in ['green','lightblue','red','pink']:
        icon = AwesomeIcon(name='fa-tint', marker_color=c, icon_color='black', spin=False)
        m.add_well([-43.51876443245584+(random()-0.5)*0.01, 
            172.66858981519297+(random()-0.5)*0.01], icon=icon)
        Q = FloatSlider(value=10, description=r'$Q$ [L/s]', min = 0, max = 40, step = 5, 
            continuous_update = False, layout=Layout(max_width='230px'),
            style = {'description_width': '60px'})
        b = Button(disabled=False,icon='fa-tint',layout=Layout(max_width='230px'))
        b.style.button_color = c
        Qs.append(Q)
        bs.append(b)  

    m.configure(widgets=dict([('Q{:d}'.format(i),Q) for i,Q in enumerate(Qs)]), 
        func=partial(m.superposition, T))
    return VBox([m, HBox([VBox([b,Q]) for Q,b in zip(Qs,bs)])])
def theis_fun():
    center=[-43.51876443245584, 172.66858981519297]
    m = GroundwaterMap(basemap=basemaps.Esri.WorldImagery, center=center, zoom=13)
    icon = AwesomeIcon(name='fa-tint', marker_color='green', icon_color='black', spin=False)
    m.add_well([-43.5187, 172.6685], icon=icon)
    Q = FloatSlider(value=25, description=r'$Q$ [L/s]', min = 0, max = 100, step = 5, 
        continuous_update = False, layout=Layout(max_width='230px'),
        style = {'description_width': '60px'})
    m.configure(widgets={'Q0':Q}, func=partial(m.superposition, 0.025))
    return VBox([m, Q])

if __name__ == "__main__":
    travel_time_fun()