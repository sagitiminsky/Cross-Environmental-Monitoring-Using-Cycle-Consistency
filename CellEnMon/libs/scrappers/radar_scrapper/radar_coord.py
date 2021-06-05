#!/usr/bin/python

"""
This code calculates radar image edge coordinates
Based on radar 280km PPI scan
written by IMS 28/11/2018, R&D
for any question contact Elyakom Vadislavsky
email: vadislavskye@ims.gov.il
"""


import time
import numpy as np
import pyproj




if __name__ == "__main__":

	start = time.time()

	Range = 280500. # [m] --> 280km

	ITM = pyproj.Proj("+init=EPSG:2039") # define ITM coords
	wgs84 = pyproj.Proj("+init=EPSG:4326") # define wgs84 coords

	Xcenter,Ycenter = pyproj.transform(wgs84,ITM,34.81456004, 32.007) # transform between coords WGS84-->ITM	

	print ("\n\n\n")
	print ("IMS RADAR Image coordinates for 280km scan")
	print ("\n")


	print ("IMS RADAR location: Latitude: 32.007 N Longitude: 34.81456004 E\n\n")

	# calculate NW point	
	Xpoint = Xcenter-Range 
	Ypoint = Ycenter+Range

	LonNW,LatNW = pyproj.transform(ITM,wgs84,Xpoint, Ypoint) # transform between coords ITM-->WGS84	

	print ("North West point Latitude:",LatNW,"N, Longitude:",LonNW,"E")

	# calculate SW point	
	Xpoint = Xcenter-Range 
	Ypoint = Ycenter-Range

	LonSW,LatSW = pyproj.transform(ITM,wgs84,Xpoint, Ypoint) # transform between coords ITM-->WGS84	

	print ("South West point Latitude:",LatSW,"N, Longitude:",LonSW,"E")

	# calculate NE point	
	Xpoint = Xcenter+Range 
	Ypoint = Ycenter+Range

	LonNE,LatNE = pyproj.transform(ITM,wgs84,Xpoint, Ypoint) # transform between coords ITM-->WGS84	

	print ("North East point Latitude:",LatNE,"N, Longitude:",LonNE,"E")

	# calculate SE point	
	Xpoint = Xcenter+Range 
	Ypoint = Ycenter-Range

	LonSE,LatSE = pyproj.transform(ITM,wgs84,Xpoint, Ypoint) # transform between coords ITM-->WGS84	

	print ("South East point Latitude:",LatSE,"N, Longitude:",LonSE,"E")

	print ("\n\n\n")

	end = time.time() # stop running time stopper
	print (end - start)

