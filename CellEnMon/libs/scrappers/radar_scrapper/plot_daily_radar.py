#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

This python code is an example how to plot RADAR products
Written by IMS 	10/12/2018
for further questions contact IMS R&D Department - Elyakom Vadislavsky
email: vadislavskye@ims.gov.il 

"""

# import python packages
import glob
import sys
import os
import numpy as np
import pandas as pd
import scipy.io
import math
import time
import datetime
import pyproj
import csv

# Graphics library

import matplotlib
matplotlib.use('Agg') # This is for runing without gui - unmark using plot show()
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl 
import matplotlib.ticker as ticker
import matplotlib.font_manager as font_manager

# import mapping library
from mpl_toolkits.basemap import Basemap, addcyclic

# parallel computing library
import multiprocessing 



def readcsv(file2read):

	with open(file2read,'r') as dest_f:
		data_iter = csv.reader(dest_f, 
			           delimiter = ',', 
			           quotechar = '"')
	   
		data = [data for data in data_iter]

	data = np.asarray(data) 

	return data



###############################################################
#  graphics functions
def encode_data_genericRR(datain,levels):

	# This function encode data in 0,1,2,3... 
	# according to levels inupt


	ll = len(levels)

	if datain < levels[0]:
		nclr = -1		

	for step in range(1,ll):
		if levels[step-1] <= datain and datain < levels[step]:
			nclr = step-1


	if levels[ll-1] <= datain:
		nclr = ll

	return nclr

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

#################################################################

# plot Image 

def plotImage(dat,fout,acc1,acc2,acc3):

	global X1,XX1,Y1,YY1
	global M1,cm1,cm2

	####################
	# load data for plot

	# draw data			

	my_dpi = 100
	fig = plt.figure(figsize=(1200/my_dpi, 1000/my_dpi), dpi=my_dpi)
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
		
	
	# precipitation levels:
	levels = [0.05,0.1,1.,4.,7.,10.,15.,20.,25.,30.,40.,50.,60.,80.,100.] 

	# define border line width
	blw = 0.5	


	############################################

	# Draw RADAR data

	plt.subplot(1,3,1)

	out = acc1

	nj,ni = np.shape(out)

	for j in range(0,nj):
		for i in range(0,ni):
				out[j,i] = encode_data_genericRR(out[j,i],levels)

	pm = M1.pcolor(X1,Y1,out,vmin=0,vmax=14)
	pm.set_cmap(cm1) # change to user colorbar

	# draw Israels border	
	M1.plot(XX1,YY1,linewidth=blw,color='k') # plot borders
	
	# draw coastlines 
	M1.drawcoastlines(linewidth=blw)
	
	# draw parallers and meridians
	# Latitude
	parallels = np.arange(29.5,34.,0.5)
	M1.drawparallels(parallels,labels=[1,0,0,1],fontsize=10,color='w')  
	# Longitude
	meridians = np.arange(34.,36.5,0.5)
	M1.drawmeridians(meridians,labels=[1,0,0,1],fontsize=10,color='w') 

	str1 = 'RR' 
	plt.title(str1,fontsize=16)
	
	############################################
	
	# Draw IDW data

	plt.subplot(1,3,2)

	out = acc2

	nj,ni = np.shape(out)

	for j in range(0,nj):
		for i in range(0,ni):
				out[j,i] = encode_data_genericRR(out[j,i],levels)

	pm = M1.pcolor(X1,Y1,out,vmin=0,vmax=14)
	pm.set_cmap(cm1) # change to user colorbar

	# draw Israels border	
	M1.plot(XX1,YY1,linewidth=blw,color='k') # plot borders

	# draw coastlines 
	M1.drawcoastlines(linewidth=blw)

	# draw parallers and meridians
	# Latitude
	parallels = np.arange(29.5,34.,0.5)
	M1.drawparallels(parallels,labels=[1,0,0,1],fontsize=10,color='w')  
	# Longitude
	meridians = np.arange(34.,36.5,0.5)
	M1.drawmeridians(meridians,labels=[1,0,0,1],fontsize=10,color='w') 	

	str1 = 'PA' 
	plt.title(str1,fontsize=16)

	############################################

	# Draw RADAR corrected data

	plt.subplot(1,3,3)

	out = acc3

	nj,ni = np.shape(out)

	for j in range(0,nj):
		for i in range(0,ni):
				out[j,i] = encode_data_genericRR(out[j,i],levels)

	pm = M1.pcolor(X1,Y1,out,vmin=0,vmax=14)
	pm.set_cmap(cm1) # change to user colorbar

	# draw Israels border	
	M1.plot(XX1,YY1,linewidth=blw,color='k') # plot borders

	# draw coastlines 
	M1.drawcoastlines(linewidth=blw)
	
	# draw parallers and meridians
	# Latitude
	parallels = np.arange(29.5,34.,0.5)
	M1.drawparallels(parallels,labels=[1,0,0,1],fontsize=10,color='w')  
	# Longitude
	meridians = np.arange(34.,36.5,0.5)
	M1.drawmeridians(meridians,labels=[1,0,0,1],fontsize=10,color='w') 

	str1 = 'RM'
	plt.title(str1,fontsize=16)

	############################################
	
	############################################

	
	cbaxes = fig.add_axes([0.14, 0.1, 0.76, 0.03]) # add_axes refer to [left, bottom, width, height]
	cbar = plt.colorbar(pm,orientation="horizontal",ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],extend='both',cax = cbaxes)  
	cbar.cmap.set_over('black')
	cbar.cmap.set_under('white')
	#cbar.set_ticklabels([0.1,1,3,5,7,10,15,20,25,30,40,50,60,80,100])
	cbar.set_ticklabels(levels)
	cbar.set_label('mm',fontsize=16)
	
	
	# add title and save file

	yyyy = dat[0:4]
	mm = dat[4:6] 
	dd = dat[6:8]
	HH = dat[8:10] 
	MM = dat[10:12]

	title1 = 'Daily (24H) Precipitation Analysis %s/%s/%s %s:%s UTC [mm]' % (dd,mm,yyyy,HH,MM)

	plt.suptitle(title1,fontsize=18)
	plt.savefig(fout,dpi=my_dpi)
	plt.close()


	plt.show()

	return 0
	############################################



def procdata(index):

	global datelist,DATADIR,PATHOUT

	dat = datelist[index] # input the Enddate

	yyyy = dat[0:4]
	mm = dat[4:6] 
	dd = dat[6:8]
	HH = dat[8:10] 
	MM = dat[10:12]

	
	# read the data
	RR = [] 
	PA = [] 
	RM = [] 
	
	f1 = '%s/RRdaily%s.asc.gz' % (DATADIR,dat)
	f2 = '%s/PAdaily%s.asc.gz' % (DATADIR,dat)
	f3 = '%s/RMdaily%s.asc.gz' % (DATADIR,dat)
	
	if os.path.isfile(f1) and os.stat(f1).st_size != 0:

		#RR = np.loadtxt(f1)
		RR = pd.read_csv(f1, sep=' ', header=None).values

	if os.path.isfile(f2) and os.stat(f2).st_size != 0:

		#PA = np.loadtxt(f2)
		PA = pd.read_csv(f2, sep=' ', header=None).values

	if os.path.isfile(f3) and os.stat(f3).st_size != 0:

		#RM = np.loadtxt(f3)
		RM = pd.read_csv(f3, sep=' ', header=None).values


	if len(RR)!=0. and len(PA)!=0. and len(RM)!=0.:
		
		fout = '%s/DailyAnalysis%s.png' % (PATHOUT,dat)
	
		plotImage(dat,fout,RR,PA,RM)
	
	return 0

########
# main #
########

if __name__ == "__main__":




	start = time.time() # start running time stopper

	# set global variables
	NX = NY = 561
	res = 1. # 1km
	RADlon,RADlat = calc_radar_lonlat(NX,NY,res)


	#######################################################################################################
	# create Basemap varaibels
	# for INCA grid
	M1 = Basemap(llcrnrlon=34.,llcrnrlat=29.3,urcrnrlon=36.,urcrnrlat=33.5,projection='mill',resolution='h')
	print ("Map on INCA grid ready !!!")


	# get RADAR map grid coords
	X1, Y1 = M1(RADlon, RADlat) # get x & y on INCA grid

	# get Borders coordinates
	# Israel borders data Base
	D = scipy.io.loadmat('MAT_files/borders_data_base.mat')
	borders = D['borders']
	XX1,YY1 = M1(borders[:,0],borders[:,1])

	######################################################

	# load color map
	H = scipy.io.loadmat('MAT_files/dbzbar.mat')
	cmR = H['dbzbar']
	ll = len(cmR)
	cm1 = cmR[1:ll-1,:]
	cm2 = cm1
	cm1 = mpl.colors.ListedColormap(cm1)


	##########################################################
	DATADIR = 'TEST_DATA' 
	PATHOUT = 'IMAGEOUT' 
	cmd = 'mkdir -p %s' % (PATHOUT)
	os.system(cmd)

	#datin = sys.argv[1] # imput date: "yyyymmdd0600", example: 201804270600
	# 0600 because data is from 06Z to 06Z
	datin = '201804270600'

	datelist = []
	datelist.append(datin)
	
	procdata(0)


	end = time.time() # stop running time stopper
	print (end - start)





