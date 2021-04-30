# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 18:18:24 2021

@author: Nilay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spatial_efd
import math
 
class FeatureExtractor:
	def __init__(self, geometry_data, edge_data, n_squamous = 20, n_columnar = 100, n_cuboidal = 10):
		if type(geometry_data) is str:
			
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
					
		self.v_id = a1
		self.vpos_x = a2
		self.vpos_y = a3
		
		if type(edge_data) is str:
			edge_data = pd.read_excel(edge_data)
			
		self.e_id = edge_data['eid']
		self.e_v1 = edge_data['v1']
		self.e_v2 = edge_data['v2']
		
		self.n_squamous = n_squamous
		self.n_columnar = n_columnar
		self.n_cuboidal = n_cuboidal
		self.n_total = self.n_squamous + self.n_columnar + self.n_cuboidal
		self.n_edges_total = 3*self.n_total
		
		
	def plot_tissue_shape(self, filename="tissue_shape.png"):
		plot1 = plt.figure(1)
		for i in range(self.n_edges_total):
			x = [self.vpos_x[self.e_v1[i]-1], self.vpos_x[self.e_v2[i]-1]]
			y = [self.vpos_y[self.e_v1[i]-1], self.vpos_y[self.e_v2[i]-1]]
			plt.plot(x,y,'black')
			
			
		plt.axes().set_aspect('equal', 'datalim')
		plt.xlabel("x [nondimensional]")
		plt.ylabel("y [nondimensional]")
		
		
		if filename is not None:
			plt.savefig(filename)
		plt.show()
		
		
		
		
		
	def edge_length(self):
		cell_edge_lengths = np.zeros(self.n_edges_total)
		id_cell_lengths = np.zeros(self.n_edges_total)
		
		for i in range(self.n_edges_total):
			x1 = self.vpos_x[self.e_v1[i]-1]
			y1 = self.vpos_y[self.e_v1[i]-1]
			x2 = self.vpos_x[self.e_v2[i]-1]
			y2 = self.vpos_y[self.e_v2[i]-1]
			
			cell_edge_lengths[i] = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
			id_cell_lengths[i] = i
			
			
		plt.plot(id_cell_lengths,cell_edge_lengths)
		plt.xlabel("Edge id")
		plt.ylabel("Edge length [nondimensional]")
		plt.show()
		
		
		return cell_edge_lengths
	
	
	
	def tissue_efd_coeff(self, harmonic):
		contour_basal_x = self.vpos_x[self.n_total:2*self.n_total-1]
		contour_basal_y = self.vpos_y[self.n_total:2*self.n_total-1]
		coeffs = spatial_efd.CalculateEFD(contour_basal_x, contour_basal_y, harmonic)
		coeffs, rotation = spatial_efd.normalize_efd(coeffs, size_invariant=True)
		xt, yt = spatial_efd.inverse_transform(coeffs, harmonic=harmonic)
		plt.plot(xt,yt,'black')
		plt.axes().set_aspect('equal', 'datalim')
		plt.xlabel("x [nondimensional]")
		plt.ylabel("y [nondimensional]")
		plt.show()
		return coeffs
	
	
	def tissue_local_curvature(self):
		contour_basal_x = self.vpos_x[self.n_total:2*self.n_total-1]
		contour_basal_y = self.vpos_y[self.n_total:2*self.n_total-1]
		
		dx = np.gradient(contour_basal_x)
		dy = np.gradient(contour_basal_y)
		
		d2x = np.gradient(dx)
		d2y = np.gradient(dy)
		
		curvature = eval('abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
		plt.plot(np.linspace(1, self.n_total-1, num=self.n_total-1),curvature)
		plt.xlabel("cell id ")
		plt.ylabel("Local curvature (1 / length)")
		plt.show()
		
		return curvature
	
	
	def tissue_basal_angle_variation(self):
		angle_neighbors = np.zeros(int(self.n_columnar / 4))
		id_angle_neighbors = np.zeros(int(self.n_columnar / 4))
		basal_center_x = self.vpos_x[self.n_total + (self.n_squamous + self.n_cuboidal + self.n_columnar)/2]
		basal_center_y = self.vpos_y[self.n_total + (self.n_squamous + self.n_cuboidal + self.n_columnar)/2]
		
		for i in range(int(self.n_columnar / 4)):
			pt1_x = self.vpos_x[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) + 1 + 2*i]
			pt1_y = self.vpos_y[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) + 1 + 2*i]
			pt2_x = self.vpos_x[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) - 1 - 2*i]
			pt2_y = self.vpos_y[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) - 1 - 2*i]
			
			v_1 = [pt1_x - basal_center_x, pt1_y - basal_center_y]
			v_2 = [pt2_x - basal_center_x, pt2_y - basal_center_y]
			
			uv_1 = v_1 / np.linalg.norm(v_1)
			uv_2 = v_2 / np.linalg.norm(v_2)
			
			dot_product = np.dot(uv_1, uv_2)
			angle_neighbors[i] = np.arccos(dot_product)
			id_angle_neighbors[i] = i
			
		plt.plot(id_angle_neighbors,angle_neighbors,'black')
		plt.show()
		
		return angle_neighbors
		
		
			