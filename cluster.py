from sklearn.cluster import KMeans
import numpy as np
import xlrd
from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import xlsxwriter
import time

def dist(a,b):
	d = 0
	for i in range(len(a)):
		if (math.isnan(a[i])==False):
			d = d + (a[i]-b[i])**2
	return d

def missing(a):
	for i in a:
		if (math.isnan(i)):
			return True
	return False

def cluster(data):
	nc = 15 #number of clusters
	kmeans = KMeans(n_clusters=nc)
	kmeans = kmeans.fit(data)
	# labels = kmeans.predict(data)
	C = kmeans.cluster_centers_
	return C

def impute_cluster(C,data):
	nc = 15
	r,c = data.shape
	data_fixed = np.zeros((r,c))
	# print (data_fixed.shape)
	for i in range(r):
		if (missing(data[i])):
			# print ("yes")
			mind = 1000000000
			cent = -1
			for j in range(nc):
				d = dist(data[i],C[j])
				if (d<mind):
					mind = d
					cent = j
			for j in range(c):
				if (math.isnan(data[i][j])):
					data_fixed[i][j] = C[cent][j]
				else:
					data_fixed[i][j] = data[i][j]
			# data_fixed[i] = C[cent]
		else:
			data_fixed[i] = data[i]

	return data_fixed












# start_time = time.time()
# data_file = ("Normdata_5000.xlsx")
# wb = xlrd.open_workbook(data_file) 
# sheet = wb.sheet_by_index(0) 

# r = sheet.nrows
# c = sheet.ncols

# data1 = np.zeros((r,c))
# for i in range(r):
# 	# print i
# 	for j in range(c):
# 		if (sheet.cell_value(i,j)!="NaN"):
# 			data1[i][j] = float(sheet.cell_value(i,j))
# 		else:
# 			data1[i][j] = float('nan')

# data = np.delete(data1, -1, 1)
# label = data1[:, -1]
# nc = 15 #number of clusters
# kmeans = KMeans(n_clusters=nc)
# kmeans = kmeans.fit(data)
# # labels = kmeans.predict(data)
# C = kmeans.cluster_centers_

# # print (C)
# t1 = time.time()

# data_file = ("data_missing.xlsx")
# wb = xlrd.open_workbook(data_file) 
# sheet = wb.sheet_by_index(0) 

# r = sheet.nrows
# c = sheet.ncols

# data1 = np.zeros((r,c))
# for i in range(r):
# 	# print i
# 	for j in range(c):
# 		if (sheet.cell_value(i,j)!="NaN"):
# 			data1[i][j] = float(sheet.cell_value(i,j))
# 		else:
# 			# print ("no")
# 			data1[i][j] = float('nan')

# data = data1

# t2 = time.time()
# # print (data[2])
# # print (float('nan'))
# # label = data1[:, -1]
# data_fixed = np.zeros((r,c))
# # print (data_fixed.shape)
# for i in range(r):
# 	if (missing(data[i])):
# 		# print ("yes")
# 		mind = 1000000000
# 		cent = -1
# 		for j in range(nc):
# 			d = dist(data[i],C[j])
# 			if (d<mind):
# 				mind = d
# 				cent = j
# 		for j in range(c):
# 			if (math.isnan(data[i][j])):
# 				data_fixed[i][j] = C[cent][j]
# 			else:
# 				data_fixed[i][j] = data[i][j]
# 		# data_fixed[i] = C[cent]
# 	else:
# 		data_fixed[i] = data[i]

# t3 = time.time()
# # print (data_fixed)
# workbook = xlsxwriter.Workbook('data_fixed2.xlsx')
# worksheet = workbook.add_worksheet()
# for i in range(np.size(data_fixed,0)):
# 	for j in range(np.size(data_fixed,1)):
# 		worksheet.write(i,j,data_fixed[i][j])

# workbook.close()