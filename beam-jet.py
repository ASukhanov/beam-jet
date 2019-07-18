#!/usr/bin/env python3
'''3D view of a beam-jet crossing. Beam (green) is horizontal along X-axis, 
jet (red) is vertical, along Z axis. Vertical sigma is calculated for the 
beam and the crossing area'''
#__version__ = 'v01 2019-07-15'#
#__version__ = 'v02 2019-07-15'# show crossection along x=0
#__version__ = 'v03 2019-07-15'# beam vSigma was wrong
#__version__ = 'v04 2019-07-17'# Camera angle is taken into account.
__version__ = 'v05 2019-07-18'# Camera is sketched on the scene

import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from timeit import default_timer as timer

#``````````````````Helper functions```````````````````````````````````````````
def quadratic_drop(x,maxX,endGain=1):
    return 1. - (1.-endGain)*(x/maxX)**2

def stDev(x,weights):
    '''returns sum, mean and standartd deviation of the weighted array 
    (histogtram)'''
    mean = np.average(x,weights=weights)
    return np.sqrt(np.average((x - mean)**2, weights=weights))
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-b','--beamSigmas',default='1,1',help=\
'Width (sigma) of the beam (X,Z), default (1,1)')
parser.add_argument('-c','--crossing',action='store_true',help=\
'Show the beam-jet crossing only (product of densities')
parser.add_argument('-j','--jetSigmas',default='1,1',help=\
'Width (sigma) of the jet (X,Y), default (1,1)')
parser.add_argument('-r','--rotate',type=float,default=45,help=\
'Rotate the crossing area by degrees in horizontal plane') 
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
w.setWindowTitle('Beam(green) sigmas:'+pargs.beamSigmas+\
', Jet(red) sigmas:'+pargs.jetSigmas)

g = gl.GLGridItem()
g.scale(10, 10, 1)
w.addItem(g)

X,Y,Z = 0,1,2
Beam,Jet = 0,1

sceneShape = 200,200,pargs.zmax # 200,200 - default of the GLVolumeItem
offset = [int(v/2) for v in sceneShape]
#scene = np.zeros(sceneShape + (4,), dtype=np.ubyte)

#``````````````````Generate beam and jet``````````````````````````````````````
def beam(ix, iy, iz, dens=(0,1,1), sigma=(0,1,1)):#, sigmaDrop=0.):
    '''Create a beam array along the axis where the sigma is  zero'''
    dens = np.array(dens)
    sigma = np.array(sigma)
    axis = np.argwhere(dens==0)[0][0]
    sigma[axis] = 1.
    invSigma = 1./sigma
    invSigma[axis] = 0.
    dens[axis] = 1.
    ddd = dens[X]*dens[Y]*dens[Z]
    ts = timer()
    x = (ix - offset[X])*pargs.cellSize
    y = (iy - offset[Y])*pargs.cellSize
    z = (iz - offset[Z])*pargs.cellSize
    #if sigmaDrop:
    #    r = np.sqrt(x**2 + y**2 + z**2)
    #    print(r.shape)
    #    invSigma *= quadratic_drop(r,1.,1.-sigmaDrop)
    v = ddd*np.exp(-0.5*((x*invSigma[X])**2 + (y*invSigma[Y])**2\
                          + (z*invSigma[Z])**2))
    print('time/cell = %.1f ns'%((timer()-ts)*1.e9/v.size))# 12ns/cell
    return v

dbeam = np.fromfunction(beam,sceneShape,dens=(0,1,1)\
       ,sigma=(0,beamSigma[0],beamSigma[1]))#,sigmaDrop=0.5)
djet = np.fromfunction(beam,sceneShape,dens=(1,1,0)\
       ,sigma=(jetSigma[0],jetSigma[1],0))
dmax = dbeam.max()
zBeamSums = np.sum(dbeam,axis=(X,Y))
bins = (np.arange(len(zBeamSums)) - offset[Z])*pargs.cellSize
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#``````````````````plot vertical profile of the beam``````````````````````````
std = stDev(bins,zBeamSums)
plt = pg.plot(title='Vertical profile',labels={'left':'Intensity [Arb.U]'\
,'bottom':'Z (vertical distance) [sigma]'})
plt.addLegend()
plt.plot(x=bins,y=zBeamSums,pen=pg.mkPen((0,200,0),width=3)\
,name='Beam, vSigma=%.2f'%std)
plt.showGrid(True,True)

#``````````````````plot vertical profile of the rotated crossing area`````````
beamJet = dbeam*djet
import scipy.ndimage.interpolation as ndimage
print('rotation',pargs.rotate)
beamJetRotated = ndimage.rotate(beamJet,pargs.rotate,(X,Y))
zCrosSums = np.sum(beamJetRotated,axis=(X,Y))
crosStDev = stDev(bins,zCrosSums)
plt.plot(bins,zCrosSums,pen=pg.mkPen((210,210,0),width=3)\
,name='Crossing, vSigma=%.2f'%crosStDev)
yCrosSums = np.sum(beamJetRotated,axis=(X,Z))
yBins = (np.arange(len(yCrosSums)) - offset[Y])*pargs.cellSize
yCrosStDev = stDev(yBins,yCrosSums)
plt.plot(yBins,yCrosSums,pen=pg.mkPen((0,0,210),width=3)\
,name='Crossing, hSigma=%.2f'%yCrosStDev)
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#``````````````````render the 3D view`````````````````````````````````````````
scene = np.zeros(dbeam.shape + (4,), dtype=np.ubyte)
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
#scene[..., 3] = dAlpha # fully colored
# right half of the scene transparent:
scene[:,:int(sceneShape[Y]/2),:,3] = dAlpha[:,:int(sceneShape[Y]/2),:]
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#``````````````````show the coordinate bars X-green, Y-blue, Z-red````````````
def bars(v):
    shape = v.shape
    v[offset[X]:offset[X]+1,offset[Y]:offset[Y]+1,:] = [255,0,0,255]
    v[:,offset[Y]:offset[Y]+1,offset[Z]:offset[Z]+1] = [0,255,0,255]
    v[offset[X]:offset[X]+1,:,offset[Z]:offset[Z]+1] = [0,0,255,255]
bars(scene)

md = gl.MeshData.cylinder(rows=10, cols=20, radius=[3, 6.0], length=20)
colors = np.ones((md.faceCount(), 4), dtype=float)
colors[::2,0] = 0
colors[:,1] = np.linspace(0, 1, colors.shape[0])
md.setFaceColors(colors)
m5 = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=True, edgeColor=(1,0,0,1), shader='balloon')
m5.rotate(90,1,0,0)
m5.rotate(-pargs.rotate,0,0,1)
if pargs.rotate > 45.:
    camX = offset[X]
    camY = offset[Y] * np.tan((90 - pargs.rotate)*np.pi/180)
else:
    camX = offset[X] * np.tan((pargs.rotate)*np.pi/180)
    camY = offset[Y]
m5.translate(camX,camY,0)
w.addItem(m5)
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

# show the scene
v = gl.GLVolumeItem(scene)
v.translate(-offset[0],-offset[1],-offset[2])
w.addItem(v)
# add axis grate
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

