import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg.basic import solve
from scipy.special import lambertw as LW
from scipy.optimize import root
from scipy.integrate import solve_ivp

from ipyleaflet import Map, basemaps, Marker, Polyline, Polygon, DivIcon
from ipywidgets import HBox, VBox, IntSlider, FloatSlider
from pyproj import Proj, transform

from functools import partial

from scipy.special import expi

inProj = Proj('epsg:32759')
outProj = Proj('epsg:4326')

class A(object):
    def __init__(self):
        pass

def W(u): 
    return -expi(-u)
def Theis(r,t,T,S,Q):
    return Q/(4*np.pi*T)*W(r**2*S/(4*T*t))

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

def travel_time(t, q, Q, n, b, theta=0., xw=0, yw=0):
    '''
        t : float
            time [days]
        q : float
            specific discharge [m/s]
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
    t*=24*3600    # convert days to seconds
    t0 = 2.*np.pi*q**2/(b*n*Q)*t
    x0,y0 = travel_time_dimensionless(t0)
    x = Q*x0/(2*np.pi*q)
    y = Q*y0/(2*np.pi*q)
    x,y = np.concatenate([x, x[::-1]]), np.concatenate([y, -y[::-1]])
    x,y = (np.cos(theta)*x-np.sin(theta)*y, np.sin(theta)*x+np.cos(theta)*y)
    return x+xw,y+yw

def test():
    f,ax = plt.subplots(1,1)
    for t0 in np.logspace(-1,1,10):
        x,y = travel_time(t0, 1., 2*np.pi, 1, 1, np.pi/4.*0.)
        ax.plot(x,y,'k-')
    x,y = travel_time(1.e7, 1.e-4, 0.1, 0.1, 10., theta=3.1416/4.)
    ax.plot(x,y,'b-')
    ax.axhline(0, color='k', linestyle=':', linewidth=0.5)
    ax.axvline(0, color='k', linestyle=':', linewidth=0.5)
    ax.set_aspect('equal')
    plt.savefig('travel_time_test.png', dpi=400)

def xy2ll(x,y,lat0,lon0):   
    x0,y0=transform(outProj, inProj, lat0, lon0)#, lat0)
    lat,lon = transform(inProj,outProj,x+x0,y+y0)
    return list(lat), list(lon)

def TheisContours(ws):
    lats = []
    lons = []
    for w in ws:
        lat,lon = w.location
        lats.append(lat); lons.append(lon)
    xs,ys = transform(outProj, inProj, lats, lons)
    n = 100
    xx,yy = np.meshgrid(np.linspace(np.min(xs)-1.e4,np.max(xs)+1.e4,n), 
        np.linspace(np.min(ys)-1.e4,np.max(ys)+1.e4,n))
    hh = 0.*xx
    for w,x,y in zip(ws,xs,ys):
        hh += Theis(np.sqrt((xx.flatten()-x)**2+(yy.flatten()-y)**2), 100.*24*3600, 0.025, 1.e-4, 0.04).reshape(xx.shape)
    lat,lon=transform(inProj,outProj,xx.flatten(),yy.flatten())
    cs = plt.contour(lat.reshape(xx.shape), lon.reshape(yy.shape), hh, levels=(0.5,0.75,1,1.25,1.5,1.75,2.0))
    ts = plt.clabel(cs)
    plt.close()
    
    allsegs = cs.allsegs
    allkinds = cs.allkinds

    ps = []
    for clev in range(len(cs.allsegs)):
        kinds = None if allkinds is None else allkinds[clev]
        segs = split_contours(allsegs[clev], kinds)
        polygons = Polygon(
                        locations=[p.tolist() for p in segs],
                        color='yellow',
                        weight=1,
                        opacity=1.,
                        fill_color=None,
                        fill_opacity=0.0
        )
        ps.append(polygons)
    return ps,ts

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

def travel_time_fun():
    x,y = travel_time(100., 1.e-4, 0.1, 0.1, 10., theta=3.1416/4.*3)
    center=[-43.51876443245584, 172.66858981519297]
    zoom=14
    m = Map(basemap=basemaps.Esri.WorldImagery, center=center, zoom=zoom)
    well = Marker(location=center, draggable=True)    
    m.add_layer(well)
    lat,lon = xy2ll(x,y,*center)
    line = Polyline(locations=list(zip(lat,lon)), color='red', fill=False, weight=2)
    m.add_layer(line)
    t = IntSlider(value=100, description=r'$t$ [day]', min=10, max = 365, step = 20, continuous_update = False)
    Q = FloatSlider(value=100, description=r'$Q$ [L/s]', min = 20, max = 200, step = 20, continuous_update = False)
    def on_change(event):
        x,y = travel_time(t.value, 1.e-4, Q.value/1.e3, 0.1, 10., theta=3.1416/4.*3)
        lat,lon = xy2ll(x,y,*well.location)
        line.locations = list(zip(lat,lon))
    Q.observe(on_change)#_Q, names='value')
    well.observe(on_change)#, names='location')
    t.observe(on_change)
    return VBox([m, HBox([Q, t])])

def superposition_fun():
    c = [-43.51876443245584, 172.66858981519297]
    m = Map(basemap=basemaps.Esri.WorldImagery, center=c, zoom=14)
    w1 = Marker(location=[-43.51876443245584, 172.66858981519297], draggable=True)    
    w2 = Marker(location=[-43.5158209576725, 172.68432984856096], draggable=True)
    m.add_layer(w1)
    m.add_layer(w2)

    ps0,ts = TheisContours([w1,w2])
    for polygons in ps0:
        m.add_layer(polygons)
    a = A()
    a.tsms=[]
    for t in ts:
        i = DivIcon(html=t._text)
        tsm = Marker(location=[t._x, t._y], icon=i)
        m.add_layer(tsm)
        a.tsms.append(tsm)

    def on_change(a, event):
        ps, ts = TheisContours([w1,w2])
        for p,p0 in zip(ps,ps0):
            p0.locations = p.locations

        for t in a.tsms:m.remove_layer(t)

        tsms=[]
        for t in ts:
            i = DivIcon(html=t._text)
            tsm = Marker(location=[t._x, t._y], icon=i)
            m.add_layer(tsm)
            tsms.append(tsm)
        a.tsms = tsms

    w1.observe(partial(on_change,a))
    w2.observe(partial(on_change,a))
    return m#VBox([m, HBox([Q, t])])

if __name__ == "__main__":
    superposition_fun()