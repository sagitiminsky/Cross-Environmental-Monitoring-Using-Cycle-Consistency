#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

This python code is an example how to read data samples from RADAR products
Written by IMS 	10/12/2018
for further questions contact IMS R&D Department - Elyakom Vadislavsky
email: vadislavskye@ims.gov.il 

"""


# import python Libraries

# for math opeartion
import math
import numpy as np
import os
import pyproj
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt


##################
#   functions    # 
##################

def get_current_rain(datin):

	global DATADIR

	
	RM = [] # case of no rain
	
	f1 = '%s/RMdaily%s.asc.gz' % (DATADIR,datin)
	
	RM = []

	if os.path.isfile(f1) and os.stat(f1).st_size != 0:

		#RM = np.loadtxt(f1)
		RM = pd.read_csv(f1, sep=' ', header=None).values

	return RM



def get_data(datin,LON,LAT):

	global missval,fout
	global wgs84,ITM
	global Xcenter,Ycenter
	

	##################################
	# input date

	yyyy = int(datin[0:4]) # get year 
	mm = int(datin[4:6]) # get month 
	dd = int(datin[6:8]) # get day 
	HH = int(datin[8:10]) # get hour 
	MM = int(datin[10:12]) # get minute 
	
	###################################
	# navigation section
	
	# Calculate Latitude & Longitude at accident location
	X,Y = pyproj.transform(wgs84,ITM,LON,LAT) # transform between coords

	# X and Y are in ITM (Israel Transverse mercator)
	# http://spatialreference.org/ref/epsg/2039/

	# Calculate RADAR's grid point location
	Xrad = int(round(((280000.+(X-Xcenter))/1000.)/1.))  
	Yrad = int(round(((280000.-(Y-Ycenter))/1000.)/1.)) 

	###################################
	# Precipitation section
	
	# get current rain data 

	RM = get_current_rain(datin)

	Rcurrent = missval

	if len(RM)!=0:
		rdata1 = RM[Yrad,Xrad]
		if rdata1 >= rain10_threshold:	
			Rcurrent = rdata1

			print ("At Lat: %2.4fN, Lon: %2.4f, the precipitation is: %2.1f" % (LAT,LON,Rcurrent))

			plotfigure(RM,Xrad,Yrad)

	return 0


def plotfigure(DATA,x,y):

	DATA[DATA<0.1] = nan

	plt.figure()
	plt.imshow(DATA)
	plt.plot(x,y,'ko',markersize=8)
	str1 = '%2.1f' % DATA[y,x]
	plt.text(x,y,str1,fontsize=14,color='red')
	plt.colorbar()
	plt.show()



###################################



########
# main #
########

if __name__ == "__main__":
	
	start = time.time()

	nan = float('NaN')
	missval = -999.
	rain10_threshold = 0.05 # 0.05 mm/hr drops at 10 minutes interval

	DATADIR = 'TEST_DATA' 

	# Navigation projection defintion

	ITM = pyproj.Proj("+init=EPSG:2039") # define ITM projection
	wgs84 = pyproj.Proj("+init=EPSG:4326") # define wgs84 projection

	Xcenter,Ycenter = pyproj.transform(wgs84,ITM,34.81456004, 32.007) # IMS RADAR location in ITM coordinates 
	  
	dat = '201804270600'
	# run the function

	lat = 31.5 # North
	lon = 35.25 # East
	get_data(dat,lon,lat) 


	end = time.time()
	print ('PROGRAM finished after %2.1f minutes' % ((end - start)/60.))


