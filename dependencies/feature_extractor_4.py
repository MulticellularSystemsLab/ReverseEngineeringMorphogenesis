# -*- coding: utf-8 -*-
"""
Code for extracting morphological features from Surface evolver simulation data

Created on Sat Apr 17 18:18:24 2021
@author: Nilay
email: nkumar4@nd.edu
Multicellular Systems Engineering Lab (MSELab)
Department of Chemical and Biomolecular Engineering
Institution: University of Notre Dame
"""

# importing libraries: Spatial efd has been loaded for extracting efd coefficients
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spatial_efd
import math
import os
 
class FeatureExtractor:
	""" 
	Arguements
		1) Vertex positions output as a result of surface evolver simulation (.txt file, 1st and 2nd columns contain x and y coordinates of the nodes in SE geometry)
	
	Returns
		1) edge_length(dtype:float, size:(n_total,1)): Cell height of each individual cell 
		2) tissue_efd_coeff: EFD approximation of the shape of cross section (4,n_harmonics))
		3) tissue_local_curvature: Local Curvature (n_total-1,1)
		4) tissue_basal_angle_variation: Angular approxinmation of curvature (n_columnar/4,1)
		5) Function plot_tissue_shape also plots and saves the tissue shape
	"""
	
	def __init__(self, geometry_data, edge_data, n_squamous = 20, n_columnar = 100, n_cuboidal = 10):
		"""Initialize with geometry and edge data
		
		Arguements
		A) geometry_data: Datafile containing geometry information,  atext file containing the vertices of nodes from Surface Evolver output file 
		B) edge_data: DataFrame or Excel filename containing edge information and its association or cell type
		C) n_squamous: Number of squamous cells (fixed = 20)
		D) n_cuboidal: Number of squamous cells (fixed = 10)
		E) n_columnar: Number of squamous cells (fixed = 100)
		"""
		# save geometry data
		if type(geometry_data) is str:
			# Checking if the file containing vertices coordinates are empty
			if os.stat(geometry_data).st_size != 0:
				a1 = []
				a2 = []
				a3 = []
				with open(geometry_data) as f:
					next(f)
					for line in f:
						data = line.split()
						a1.append(float(data[0]))
						a2.append(float(data[1]))
						a3.append(float(data[2]))
						
			else:
				a1 = 0
				a2 = 0
				a3 = 0
		# Saving the x and y coordinates of the nodes og wing disc geometry from surface evolver (SE)			
		self.v_id = a1
		self.vpos_x = a2
		self.vpos_y = a3
		
		# save edge data
		if type(edge_data) is str:
			edge_data = pd.read_excel(edge_data)
			
		self.e_id = edge_data['eid']
		self.e_v1 = edge_data['v1']
		self.e_v2 = edge_data['v2']
		
		# Define geometrical attributes
		self.n_squamous = n_squamous
		self.n_columnar = n_columnar
		self.n_cuboidal = n_cuboidal
		self.n_total = self.n_squamous + self.n_columnar + self.n_cuboidal
		self.n_edges_total = 3*self.n_total
		
		
	def plot_tissue_shape(self, filename="tissue_shape.png"):
		
		""" Plotting the disc shape using vertex and edge data
		
		Arguements
			1) filename: If string, save plot to file. If None, skip
			
		Returns
			1) Nothing
			
		Action 
			1) Creates plot
		"""
		plot1 = plt.figure(1)
		
		# loop over all coordinates
		for i in range(self.n_edges_total):
			# assemble edges
			x = [self.vpos_x[self.e_v1[i]-1], self.vpos_x[self.e_v2[i]-1]]
			y = [self.vpos_y[self.e_v1[i]-1], self.vpos_y[self.e_v2[i]-1]]
			# plot edge
			plt.plot(x,y,'black')
			
		# Adjust and label axis
		plt.axes().set_aspect('equal', 'datalim')
		plt.xlabel("x [nondimensional]")
		plt.ylabel("y [nondimensional]")
		
		
		if filename is not None:
			plt.savefig(filename)
		plt.show()
		
		
	def edge_length(self):
		""" Extracting edge length for each cell
		
		1) Arguements
		
		2) Returns
			An array containing edge length of each cells
			
		3) Action
			Estimation of tissue geometrical property, cell length, for each individual cell in the model
		"""
		
		# Initializing the arrays for storing edge lengths
		cell_edge_lengths = np.zeros(self.n_edges_total)
		id_cell_lengths = np.zeros(self.n_edges_total)
		# Checking if the file containing vertices is not empty
		if self.vpos_x != 0:
		# Looping over all the cell edges
			for i in range(self.n_edges_total):
				
				# Getting x and y co-ordinates of each nodes in the edge
				x1 = self.vpos_x[self.e_v1[i]-1]
				y1 = self.vpos_y[self.e_v1[i]-1]
				x2 = self.vpos_x[self.e_v2[i]-1]
				y2 = self.vpos_y[self.e_v2[i]-1]
				
				# Calculating eucliean distance between the point
				cell_edge_lengths[i] = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
				id_cell_lengths[i] = i
		
		# Passing each edge length of model output as a return value of this function
		return cell_edge_lengths
	
	
	
	def tissue_efd_coeff(self, harmonic):
		"""EFD fit for the whole tissue geometry
		
		Arguewmwnts
			1) harmonic (int, scalar): Number of harmonics or terms in efd estimation
			
		Returns
			1) coeffs (float, 4 x n_harmonics): n array containing EFD coefficients normalized against size
	      2) coeffs_norm (float, 4 x n_harmonics): an array containing EFD coefficients normalized against orientation and size
	      3) rotation (float, scalar): The angle as a reasults of rotation during normalization
			
		Action
			1) Extraction of the EFD based rotation and translation invariant shape descriptors
		"""
		# Checking if the file cobtaining the vertices is not empty
		if self.vpos_x != 0:
		# # Extracting the nodes located in the basal surface of tissue (both squamous and columnar)
			contour_basal_x = self.vpos_x[self.n_total:2*self.n_total-1]
			contour_basal_y = self.vpos_y[self.n_total:2*self.n_total-1]
			
			# Using spatial efd library to extract efd coefficients from tissue boundary points
			coeffs = spatial_efd.CalculateEFD(contour_basal_x, contour_basal_y, harmonic)
			# Normalizing the coefficients against rotation and size
			coeffs_norm, rotation = spatial_efd.normalize_efd(coeffs, size_invariant=True)
			
			# Reverse EFD for plotting the normalized tissue shape
			xt, yt = spatial_efd.inverse_transform(coeffs_norm, harmonic=harmonic)
		else:
			coeffs = 0
			coeffs_norm = 0

		# Passing coefficient as a return value of this function
		return coeffs, coeffs_norm, rotation
	
	
	def tissue_local_curvature(self):
		""" Extracts local tissue curvature along the exterior basal surface
		
		Arguements
		
		Returns
			1) an array containing local tissue curvature along the basal surface
			
		Action
			2) The functions measures local tisue curvature along the basal surface
	         Curvature is extimated using the formulae defined in the eval input arguement
		"""
		# Checking if the file containing vertices is empty or not
		if self.vpos_x != 0:
			# Extracting the nodes located in the basal surface of tissue (both squamous and columnar)
			contour_basal_x = self.vpos_x[self.n_total:2*self.n_total-1]
			contour_basal_y = self.vpos_y[self.n_total:2*self.n_total-1]
			
			# Calculating first derivative
			dx = np.gradient(contour_basal_x)
			dy = np.gradient(contour_basal_y)
			
			# Calculating second derivative
			d2x = np.gradient(dx)
			d2y = np.gradient(dy)
			
			# Calculating curvature
			curvature = eval('abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
		else:
			curvature = 0
		
		# Returning curvature values
		return curvature
	
	
	def tissue_basal_angle_variation(self):
		"""Extracting tpoplogical information about the columnar-basal surface of epitheila:
			cells at equual distances from the center are selected for increasing distance
			from the center and the angle subtended by these points is measured around center
			
		Arguements
			
		Returns
			variation of angle formed by cells located at equal distances from the center is
			returned as an output
			
		Action
			Center of the basal epithelia is located
			Nodes of cells at varying distance from the center is calculated
			Angle between the three points is measure
		"""
		
		# Arrays are initialized
		angle_neighbors = np.zeros(int(self.n_columnar / 4))
		id_angle_neighbors = np.zeros(int(self.n_columnar / 4))
		
		# Center of the basement membrane is calculated
		basal_center_x = self.vpos_x[self.n_total + (self.n_squamous + self.n_cuboidal + self.n_columnar)/2]
		basal_center_y = self.vpos_y[self.n_total + (self.n_squamous + self.n_cuboidal + self.n_columnar)/2]
		
		# Nodes are selected at varying sistance from the ceneter
		for i in range(int(self.n_columnar / 4)):
			
			# x and y coordinates of the nodes are calculated
			pt1_x = self.vpos_x[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) + 1 + 2*i]
			pt1_y = self.vpos_y[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) + 1 + 2*i]
			pt2_x = self.vpos_x[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) - 1 - 2*i]
			pt2_y = self.vpos_y[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) - 1 - 2*i]
			
			# Line koining these nodes and the center are defined as vectors
			v_1 = [pt1_x - basal_center_x, pt1_y - basal_center_y]
			v_2 = [pt2_x - basal_center_x, pt2_y - basal_center_y]
			
			# Unit vectors are defined
			uv_1 = v_1 / np.linalg.norm(v_1)
			uv_2 = v_2 / np.linalg.norm(v_2)
			
			# Angle between the vectors is measured
			dot_product = np.dot(uv_1, uv_2)
			angle_neighbors[i] = np.arccos(dot_product)
			id_angle_neighbors[i] = i
		
		# plotting the variation in angle
		plt.plot(id_angle_neighbors,angle_neighbors,'black')
		plt.show()
		
		# returning the feature vector
		return angle_neighbors