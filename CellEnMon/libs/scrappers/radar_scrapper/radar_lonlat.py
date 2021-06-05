
"""
	This code calculates IMS radar lon/lat gridpoints
	written by IMS: 09/01/2019
	for questions please contact Elyakom Vadislavsky, IMS R&D,
	email: vadislavskye@ims.gov.il
"""


import numpy as np
import pyproj
import matplotlib.pyplot as plt

def calc_radar_lonlat(NX,NY,res):

	ITM = pyproj.Proj("+init=EPSG:2039") # define ITM coords
	wgs84 = pyproj.Proj("+init=EPSG:4326") # define wgs84 coords

	center_lat = 32.007 # IMS radar location latitude
	center_lon = 34.81456004 # IMS radar location longitude

	XCRADAR,YCRADAR = pyproj.transform(wgs84,ITM,center_lon,center_lat) # transform between coords	

	nx = (NX-1)/(2/(1./res))
	ny = (NY-1)/(2/(1./res))

	x = np.linspace(-nx, nx, NX)
	y = np.linspace(ny, -ny, NY)
	
	xv, yv = np.meshgrid(x, y)

	xv = xv*1000.
	yv = yv*1000.

	RADX = XCRADAR+xv
	RADY = YCRADAR+yv

	RADlon,RADlat = pyproj.transform(ITM,wgs84,RADX,RADY) # transform between coords


	return RADlon,RADlat


def plotfigure(data):

	plt.figure()
	plt.imshow(data)
	plt.colorbar()
	plt.show()

	return 0



# set global variables
NX = NY = 561
res = 1. # 1km
RADlon,RADlat = calc_radar_lonlat(NX,NY,res)

plotfigure(RADlon)
plotfigure(RADlat)

np.savetxt('RADlon.asc.gz',RADlon,fmt='%2.4f')
np.savetxt('RADlat.asc.gz',RADlat,fmt='%2.4f')





