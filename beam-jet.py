#!/usr/bin/env python3
'''3D view of a beam-jet crossing. Beam (green) is horizontal, 
jet (red) vertical. Vertical sigma calculated for the beam and the crossing section'''
#__version__ = 'v01 2019-07-15'#

import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from timeit import default_timer as timer

import argparse
parser = argparse.ArgumentParser(description=__doc__)
#parser.add_argument('-d','--dbg', action='store_true', help='debugging')
parser.add_argument('-c','--crossing',action='store_true',help=\
'Show the beam-jet crossing only (product of densities')
parser.add_argument('-b','--beamSigmas',default='1,1',help=\
'Width (sigma) of the beam (X and Z), default (1,1)')
parser.add_argument('-j','--jetSigmas',default='1,1',help=\
'Width (sigma) of the jet (X,Z), default (1,1)')
parser.add_argument('-s','--cellSize',type=float,default=0.05,help=\
'Size of the elementary cell, default 0.05')
parser.add_argument('-z','--zmax',type=int,default=200,help=\
'Height (Z-size of the scene, default=200, X and Y are fixed at 200')
pargs = parser.parse_args()
beamSigma = [float(i) for i in pargs.beamSigmas.split(',')]
jetSigma = [float(i) for i in pargs.jetSigmas.split(',')]

import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 200
w.show()
w.setWindowTitle(__doc__)

g = gl.GLGridItem()
g.scale(10, 10, 1)
w.addItem(g)

X,Y,Z = 0,1,2
Beam,Jet = 0,1

sceneShape = 200,200,pargs.zmax # 200,200 - default of the GLVolumeItem
offset = [int(v/2) for v in sceneShape]
scene = np.zeros(sceneShape + (4,), dtype=np.ubyte)

def beam(ix, iy, iz, dens=(0,1,1), sigma=(0,1,1)):
    '''Create a beam array along the axis where sigmama is  zero'''
    dens = np.array(dens)
    sigma = np.array(sigma)
    axis = np.argwhere(dens==0)[0][0]
    dens[axis] = 1.
    ddd = dens[X]*dens[Y]*dens[Z]
    sigma[axis] = 1.
    s = 1./sigma
    s[axis] = 0.
    ts = timer()
    x = (ix - offset[X])*pargs.cellSize
    y = (iy - offset[Y])*pargs.cellSize
    z = (iz - offset[Z])*pargs.cellSize
    v = ddd*np.exp(-0.5*((x*s[X])**2 + (y*s[Y])**2 + (z*s[Z])**2))
    #print('time/cell = %.1f ns'%((timer()-ts)*1.e9/v.size))# 12ns/cell
    return v

dbeam = np.fromfunction(beam,scene.shape[:3],dens=(0,1,1)\
       ,sigma=(0,beamSigma[0],beamSigma[1]))
djet = np.fromfunction(beam,scene.shape[:3],dens=(1,1,0)\
       ,sigma=(jetSigma[0],jetSigma[1],0))
dmax = dbeam.max()
yBeamSums = np.sum(dbeam,axis=(X,Z))
bins = np.arange(len(yBeamSums))*pargs.cellSize

def stDev(x,weights):
    mean = np.average(x,weights=weights)
    return np.sqrt(np.average((x - mean)**2, weights=weights))
std = stDev(bins,yBeamSums)
plt = pg.plot(title='Vert. beam profile')
plt.addLegend()
plt.plot(x=bins,y=yBeamSums,pen='k',name='Beam, vSigma=%.2f'%std)
plt.showGrid(True,True)

zCrosSums = np.sum(dbeam*djet,axis=(X,Y))
crosStDev = stDev(bins,zCrosSums)
plt.plot(bins,zCrosSums,pen='b',name='Crossing, vSigma=%.2f'%crosStDev)

scene = np.empty(dbeam.shape + (4,), dtype=np.ubyte)

dBlue = 0
if pargs.crossing:
    dRed = dbeam * djet
    dRed = 255.*dRed/dRed.max()
    dGreen = dRed
    dAlpha = ((dRed*0.5 + dGreen*0.5).astype(float) / 255.)**2 * 255
else:
    g = 255./dbeam.max(),255./djet.max() 
    dRed = g[Jet]*djet
    dGreen = g[Beam]*dbeam
    dAlpha = ((dRed*0.5 + dGreen*0.5).astype(float) / 255.)**2 * 255

scene[..., 0] = dRed
scene[..., 1] = dGreen
scene[..., 2] = dBlue
scene[..., 3] = dAlpha

def bars(v):
    shape = v.shape
    v[0:2,0:2,:] = [255,0,0,255]
    v[:,0:2,0:2] = [0,255,0,255]
    v[0:2,:,0:2] = [0,0,255,255]
bars(scene)    

v = gl.GLVolumeItem(scene)
v.translate(-offset[0],-offset[1],-offset[2])
w.addItem(v)

ax = gl.GLAxisItem()
w.addItem(ax)

# enable Ctrl-C to kill application
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)    

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

