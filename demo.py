import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from ipyleaflet import Map, basemaps, Marker, Polygon, DivIcon, AwesomeIcon
from ipywidgets import HBox, VBox, FloatSlider, Button, Layout, FloatProgress, Label
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
    def add_well(self, location, icon=None):
        well = Marker(location=location, draggable=True, icon=icon)    
        self.add_layer(well)
        self.wells.append(well)
    def update_contours(self, ps, ts):
        for p,p0 in zip(ps,self.ps0):
            p0.locations = p.locations

        if self.drawdown_labels:
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
    def superposition(self, T, bnds=None):
        qs = [self.widgets['Q{:d}'.format(i)].value/1.e3 for i,w in enumerate(self.wells)]

        # add or update piezometric surface
        if bnds is None:
            bnds = self.bounds
            if len(bnds) == 0:
                
                bnds = ((self.center[0]-0.35, self.center[1]-0.082), 
                    (self.center[0]+0.35, self.center[1]+0.082))
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
        colors[-1]='#%02x%02x%02x' % tuple(int(j*255) for j in [1,0,0])
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

def get_style(val):
    if val>0.8:
        return 'success'
    elif val>0.6:
        return 'info'
    elif val>0.4:
        return 'warning'
    else:
        return 'danger'

def placing_groundwater_wells(transmissivity=0.025, center=[-43.519, 172.669], drawdown_labels=False):
    m=GroundwaterMap(basemap=basemaps.Esri.WorldImagery, center=center, zoom=13,
        layout=Layout(height='500px'))
    m.drawdown_labels=drawdown_labels
    m.center=center
    Qs = []
    bs = []
    maxQ=40
    Q0=20
    for i,c in enumerate(['green','lightblue','red']):
        icon = AwesomeIcon(name='fa-tint', marker_color=c, icon_color='black', spin=False)
        m.add_well([center[0]+(random()-0.5)*0.05, 
            center[1]+(random()-0.5)*0.05], icon=icon)
        Q = FloatSlider(value=Q0, description=r'Well {:d}'.format(i+1), min = 0, max = maxQ, step = 5, 
            continuous_update = False, layout=Layout(max_width='180px'), style = {'description_width': '60px'},
            readout=False)
        b = Button(disabled=False,icon='fa-tint',layout=Layout(max_width='250px'))
        b.style.button_color = c
        Qs.append(Q)
        bs.append(b)  
    lbl=Label(value="Water Volume Obtained", layout=Layout(width='80%', display='flex' ,
    align_items='center', justify_content="flex-end"))
    fp=FloatProgress(value=(i+1)*Q0,min=0,max=(i+1)*maxQ,step=5,description='{:2d}%'.format(int(Q0/maxQ*100)),
        readout=True, bar_style=get_style(Q0/maxQ))
    def total_change(change):
        fp.value=np.sum([Q.value for Q in Qs])
        fp.description='{:2d}%'.format(int(fp.value/fp.max*100))
        fp.bar_style=get_style(fp.value/fp.max)
    [Q.observe(total_change, names = 'value') for Q in Qs]
    
    m.configure(widgets=dict([('Q{:d}'.format(i),Q) for i,Q in enumerate(Qs)]), 
        func=partial(m.superposition, transmissivity))
       
    return VBox([m, HBox([VBox([fp,lbl])]+[VBox([b,Q]) for Q,b in zip(Qs,bs)])])

if __name__ == "__main__":
    placing_groundwater_wells()