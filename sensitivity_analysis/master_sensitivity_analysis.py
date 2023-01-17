# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 23:35:16 2021

@author: Nilay Kumar
"""
# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import spatial_efd
import math 
import signac
import numpy as np
import os.path
import os
import torch
import gpytorch
import subprocess
import gc
import similaritymeasures
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from smt.sampling_methods import LHS

# Importing helper libraries for bayesian optimization
from dependencies.data_preprocessing_class import dataPreprocessing
from dependencies.gaussian_process_regression_class import gaussianProcessRegression
from dependencies.acquisition_functions_class import acqisitionFunctions
from dependencies.geometry_writer import geometryWriter
from dependencies.feature_extractor_4 import FeatureExtractor

"""
STEP 1: Reading in the target shape data (point around which sensitivity has to be calculated)
"""
# Reading the vertices output file from a sample SE simulation output with known parameters
fe_exp = FeatureExtractor('input_data/vertices_target_SE.txt', 'log_edges.xlsx')
# Extracting the efd coefficients
dummy11, coeffs_exp_basal, dummy12, dummy21, coeffs_exp_apical, dummy22 = fe_exp.tissue_efd_coeff(20)
# Obtaining normalized x and y coordinates for the apical surface of the tissue
xt_exp_apical, yt_exp_apical = spatial_efd.inverse_transform(coeffs_exp_apical, harmonic=20)
# Initializing the array that stores the x and y coordinates of the equilibrium apical shape around which sensitivity is calculated
exp_data_apical = np.zeros((300,2))
exp_data_apical[:,0] = xt_exp_apical
exp_data_apical[:,1] = yt_exp_apical
# Obtaining normalized x and y coordinates for the basal surface of the tissue
xt_exp_basal, yt_exp_basal = spatial_efd.inverse_transform(coeffs_exp_basal, harmonic=20)
# Initializing the array that stores the x and y coordinates of the equilibrium basal shape around which sensitivity is calculated
exp_data_basal = np.zeros((300,2))
exp_data_basal[:,0] = xt_exp_basal
exp_data_basal[:,1] = yt_exp_basal


"""
STEP 2: Sensitivity analysis
"""
# Number of parameters varied in LHS out of total 35 model parameter
n_param_model = 22
# Indices of the parameters varied during the LHS
param_index_model = [1,4,7,11,12,13,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30,33]
# initializing the error array for storing the frechet errors
error_target_sampled_apical = []
error_target_sampled_basal = []
# Initializing the array for storing the parameter values
param_sampled = np.zeros((n_param_model*2,35)) # there are total 35 parameters in the model
curvature_basal_master = np.zeros((n_param_model*2,129)) #total 129 basal nodes
tissue_edge_length_master = np.zeros((n_param_model*2,390)) # total 390 edges within the model

# Counter for iterations
k = 0

for i in range(n_param_model):	
	""" 
	Step 2A
	"""
	# Parameter values for the target/equilibrium shape
	paraminputs = [0,3.162277660168379e-08,0,0,3.162277660168379e-08,0,0,1e-05,0,0,0, 0.1,0.1,10,0.1,0.1,0.1,0.01, 3.9810717055349695e-05, 0.01,0.001,0.001, 1,1,0.6,0.6,0.6,0.6,0.2,0.1,3,0.6,1.8, 0.002511886431509582,0.001]
	# Calculating the upper and lower bound for ith parameter of interest by going up and down 70% of the set value
	# param_upper_bound = param_i + 70% of value of [param_i] 
	param_upper_bound = paraminputs[param_index_model[i]] + 0.7*paraminputs[param_index_model[i]]
	# param_lower_bound = param_i - 70% of value of [param_i] 
	param_lower_bound = paraminputs[param_index_model[i]] - 0.7*paraminputs[param_index_model[i]]
	# Creating an array containing the lower and upper bounds for the sensitivity analysis
	param_sens_bound = [param_lower_bound, param_upper_bound]
	
	for j in range(2):
		""" 
		STEP 2B : Defining parameters for the sensitivity analysis
		"""
		# Setting the param of interest iteratively as the lower and upper bounds
		paraminputs[param_index_model[i]] = param_sens_bound[j]
		# Saving the parameter values
		param_sampled[k,:] = paraminputs
		
		
		"""
		STEP 2B :  Running surface evolver and extracting shape information
		"""
		# Defining the set system pressure
		param_pressure = 0.001
		# Writing geometry file
		geometryWriter(paraminputs, param_pressure, 'wingDisc')
		# Running surface evolver simulations, Can be replaced using python subprocess
		os.system("/home/nkumar4/Desktop/evolver_installation/src/evolver wingDisc.fe")
		# Calling in the feature extraction class for the surface evolver output stored in "vertices.txt"
		fe = FeatureExtractor('vertices.txt', 'log_edges.xlsx')
		# Extracting EFD coefficients of apical and basal surface of the output simulation
		dummy11, efd_coeff_sampled_basal, dummy12, dummy21, efd_coeff_sampled_apical, dummy22 = fe.tissue_efd_coeff(20)
		# Extracting x and y coordinated from the normalized apical and basal surfaces
		xt_sampled_apical, yt_sampled_apical = spatial_efd.inverse_transform(efd_coeff_sampled_apical, harmonic=20)
		xt_sampled_basal, yt_sampled_basal = spatial_efd.inverse_transform(efd_coeff_sampled_basal, harmonic=20)
		# Extracting curvature deature of the whole tissue
		curvature_basal_sampled = fe.tissue_local_curvature()
		# Extracting curvature of the basal surface adhering to the columnar cells of the pouch
		curvature_basal_sampled_reshaped = np.reshape(curvature_basal_sampled, (1, 129))
		# Storing the curvature data in the master data
		curvature_basal_master[k,:] = curvature_basal_sampled_reshaped
		# Extracting length of each edge within the SE model
		tissue_edge_length_sampled = fe.edge_length()
		tissue_edge_length_sampled_reshaped = np.reshape(tissue_edge_length_sampled, (1, 390))
		# # Storing the edge length data in the master array
		tissue_edge_length_master[k,:] = tissue_edge_length_sampled_reshaped
		
		# Moving the counter forward
		k = k + 1
		
		"""
		STEP 2C: Plotting the sampled and target shape
		"""
		# Defining filename for plot showing overlap between the sampled shape and the target shape
		filename_shape_plot_apical = str(i) + str(j) + "apical_sapled_target_xy_plot.svg"
		filename_shape_plot_basal = str(i) + str(j) + "basal_sapled_target_xy_plot.svg"
		# Plotting the equilibrium data
		plt.plot(xt_exp_apical,yt_exp_apical,'black', label='Target')
		# Plotting sampled data
		plt.plot(xt_sampled_apical, yt_sampled_apical,'blue', label='Sampled')
		# Setting aspect ratio of the plot as 1
		plt.axes().set_aspect('equal', 'datalim')
		# Labeling axes
		plt.xlabel("x [nondimensional]")
		plt.ylabel("y [nondimensional]")
		# Plotting legends
		plt.legend()
		# Saving the shape file
		plt.savefig("contour_evolution_plots/" + filename_shape_plot_apical)
		# Closing the plot
		plt.close()
		
		# Repeatig for the basal contour
		plt.plot(xt_exp_basal,yt_exp_basal,'black', label='Target')
		plt.plot(xt_sampled_basal, yt_sampled_basal,'blue', label='Sampled')
		plt.axes().set_aspect('equal', 'datalim')
		plt.xlabel("x [nondimensional]")
		plt.ylabel("y [nondimensional]")
		plt.legend()
		plt.savefig("contour_evolution_plots/" + filename_shape_plot_basal)
		plt.close()
		
		"""
		STEP 2D: Caculating frechet error between equilibrium and sampled shape
		"""
		# Initializing array to store x and y coordinates of the apical and basal surface
		sampled_data_apical = np.zeros((300,2))
		sampled_data_basal = np.zeros((300,2))
		# Stroing the x and y cpoordinates of the normalized apical surface of the sampled data
		sampled_data_apical[:,0] = xt_sampled_apical
		sampled_data_apical[:,1] = yt_sampled_apical
		# Storing the x and y cpoordinates of the normalized basal surface of the sampled data
		sampled_data_basal[:,0] = xt_sampled_basal
		sampled_data_basal[:,1] = yt_sampled_basal
		# Using the package similarityMeasures to evalueate Frechet distances of apical and basal surface of the samples dhape wrt the equilibriun shape
		error_target_sampled_step_apical = similaritymeasures.frechet_dist(exp_data_apical,sampled_data_apical)
		error_target_sampled_step_basal = similaritymeasures.frechet_dist(exp_data_basal,sampled_data_basal)
		# Appending the calculated error to the master array
		error_target_sampled_apical.append(error_target_sampled_step_apical)
		error_target_sampled_basal.append(error_target_sampled_step_basal)
		
		# Saving vertices 
		command_save_vertices = "cp vertices.txt vertices_" + str(i) + "_" + str(j) + ".txt"
		os.system(command_save_vertices)
		# Deleting unnecesssary surface evolver files
		os.system("rm vertices.txt")
		os.system("rm energylog.txt")
		os.system("rm specificenergylog.txt")
		# Deleting variables that willbe generated in the next loop to avoid overwiting (just in case :P)
		del xt_sampled_apical
		del yt_sampled_apical
		del xt_sampled_basal
		del yt_sampled_basal
		del fe
		gc.collect()
		

"""Saving important arrays
"""	
# Frechet error between the apical surfaces of the equilibrium shape and perturbed shapes 
np.save('error_target_sampled_apical.npy',error_target_sampled_apical)
# Frechet error between the basal surfaces of the equilibrium shape and perturbed shapes 
np.save('error_target_sampled_basal.npy',error_target_sampled_basal)
# Basal curvature of the columnar cells in the pouch for the perturbed shapes
np.save('curvature_basal_master.npy',curvature_basal_master)
# Edge lengths of all the cells in the model
np.save('tissue_edge_length_master.npy',tissue_edge_length_master)
# The sampled parameter sets diring the sensitivity analysis
np.save('param_sampled.npy',param_sampled)

