#!/usr/bin/env python3
'''Beam-jet interaction'''
import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from timeit import default_timer as timer

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 200
w.show()
w.setWindowTitle(__doc__)

#b = gl.GLBoxItem()
#w.addItem(b)
g = gl.GLGridItem()
g.scale(10, 10, 1)
w.addItem(g)

X,Y,Z = 0,1,2
CellSize = 0.05 # mm
def beamOld(ix, iy, iz, dens=(1.,1.), sig=(1.,1.), offset=None):
    if offset is None:
        offset = ix/2, iy/2, iz/2
    x = (ix - offset[0])*CellSize
    y = (iy - offset[1])*CellSize
    z = (iz - offset[2])*CellSize
    return dens[X]*dens[Y]*np.exp(-0.5*((x/sig[X])**2 + (y/sig[Y])**2))*10
    
def vsum(ix, iy, iz, dens=(1.,1.), sig=(1.,1.), offset=None):
    if offset is None:
        offset = ix/2, iy/2, iz/2
    x = (ix - offset[0])*CellSize
    y = (iy - offset[1])*CellSize
    z = (iz - offset[2])*CellSize
    return x+y+z

sceneShape = 200,200,100 # 200,200 - default of the GLVolumeItem
#sceneShape = 100,50,100 # 
offset = [int(v/2) for v in sceneShape]
print('offset',offset)
scene = np.zeros(sceneShape + (4,), dtype=np.ubyte)

def beam_slow(sc,dens,sig,rgba=(255,0,0,50)): #axis=X,
    shape = sc.shape[:3]
    #ofsZ = offset[Z]
    #shape = sc.shape[X], sc.shape[Y], int(sc.shape[Z]/4)
    #ofsZ = offset[Z] - shape[Z]
    dens = np.array(dens)
    sig = np.array(sig)
    print('dens',dens)
    axis = np.argwhere(dens==0)[0][0]
    print('axis',axis)
    dens[axis] = 1.
    ddd = dens[X]*dens[Y]*dens[Z]
    sig[axis] = 1.
    s = 1./sig
    s[axis] = 0.
    print(dens,s)
    ts = timer()
    for ix,iy,iz in np.ndindex(shape):
        x = (ix - offset[X])*CellSize
        y = (iy - offset[Y])*CellSize
        z = (iz - offset[Z])*CellSize
        v = ddd*np.exp(-0.5*((x*s[X])**2 + (y*s[Y])**2 + (z*s[Z])**2))
        #lrgba = [v/ddd*255,0,0,v/ddd*50]
        lrgba = [v/ddd*g for g in rgba]
        sc[ix,iy,iz] = sc[ix,iy,iz] + lrgba
    print('time/point = %.1f us'%((timer()-ts)/ix/iy/iz*1.e6)) #9.2us/cell

def beam(ix, iy, iz, dens=(0,1,1), sig=(0,1,1)):
    dens = np.array(dens)
    sig = np.array(sig)
    #print('dens',dens)
    axis = np.argwhere(dens==0)[0][0]
    #print('axis',axis)
    dens[axis] = 1.
    ddd = dens[X]*dens[Y]*dens[Z]
    sig[axis] = 1.
    s = 1./sig
    s[axis] = 0.
    #print(dens,s)
    ts = timer()
    x = (ix - offset[X])*CellSize
    y = (iy - offset[Y])*CellSize
    z = (iz - offset[Z])*CellSize
    #print('x,y,z',x.shape,y.shape,z.shape)
    v = ddd*np.exp(-0.5*((x*s[X])**2 + (y*s[Y])**2 + (z*s[Z])**2))
    print('time/cell = %.1f ns'%((timer()-ts)*1.e9/v.size))
    return v
    
#beam(scene,(0,1,1),(0,1,1),rgba=(0,127,0,50))
#print('beam',scene.shape,scene.max())
#beam(scene,(1,1,0),(1,1,0),rgba=(0,127,0,50))
#print('jet',scene.shape,scene.max())
#print(scene)

dbeam = np.fromfunction(beam,scene.shape[:3],dens=(0,1,1),sig=(0,1,1))
djet = np.fromfunction(beam,scene.shape[:3],dens=(1,1,0),sig=(1,1,0))
dmax = dbeam.max()
print('beam',dbeam.shape,dmax)
scene = np.empty(dbeam.shape + (4,), dtype=np.ubyte)

#scene[..., 0] = djet * (255./dmax)
#scene[..., 1] = dbeam * (255./dmax)
#scene[..., 2] = 0#scene[...,1]

d = (dbeam * djet)
#d = dbeam

d *= (255./d.max())
scene[..., 0] = d
scene[..., 1] = d
scene[..., 2] = d
scene[..., 3] = scene[..., 0]*0.3 + scene[..., 1]*0.3
scene[..., 3] = (scene[..., 3].astype(float) / 255.) **2 * 255

def bars(v):
    shape = v.shape
    v[0         ,0,:] = [0,255,0,255]
    v[shape[0]-1,0,:] = [0,0,255,255]
    v[0         ,shape[1]-1,:] = [255,255,0,255]
bars(scene)    

v = gl.GLVolumeItem(scene)
v.translate(-offset[0],-offset[1],-offset[2])
w.addItem(v)

ax = gl.GLAxisItem()
w.addItem(ax)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

