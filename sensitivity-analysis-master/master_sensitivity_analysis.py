# -*- coding: utf-8 -*-
"""
Code Summary: The code is used to generate the data for sensitivity analysis of parameters.
The parametsrs in  a user defined list are perturbed by increasing and decreasing them of 
the 70% of the value. The results are stored and later used for further analysis. 

The inputs in the particular code were used to generate Figure 2C within teh manuscript.
Refer sensitivity_analysis.ipynb for the analysis part


Created on Tue Jul  6 23:35:16 2021

@author: Nilay Kumar
email: nkumar4@nd.edu
Multicellular Systems Engineering Lab (MSELab)
Department of Chemical and Biomolecular Engineering
Institution: University of Notre Dame
"""
# Adding dependencies folder to the path. Dependencies stores all teh classes used in bayesian optimization BO
import sys
sys.path.append("/home/nkumar4/Desktop/")

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
from dependencies.data_preprocessing_class import DataPreprocessing
from dependencies.gaussian_process_regression_class import GaussianProcessRegression
from dependencies.acquisition_functions_class import AcqisitionFunctions
from dependencies.geometry_writer import GeometryWriter
from dependencies.feature_extractor_4 import FeatureExtractor

"""
User derived inputs
"""
# Number of parameters varied in LHS
n_param_model = 22
# Indices of the parameters varied during the LHS
param_index_model = [1,4,7,11,12,13,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30,33]
# Parameter used to define pressure in the wing imaginal disc
param_pressure = 0.001
# The model parameter values around which sensitibity is computed
paraminputs_stable = [0,3.162277660168379e-08,0,0,3.162277660168379e-08,0,0,1e-05,0,0,0, 0.1,0.1,10,0.1,0.1,0.1,0.01, 3.9810717055349695e-05, 0.01,0.001,0.001, 1,1,0.6,0.6,0.6,0.6,0.2,0.1,3,0.6,1.8, 0.002511886431509582,0.001]
# Total number of harmonics for calculation of EFD coefficients
num_harmonic_efd = 20
# name of teh file that we want to save the SE model input file as
se_filename = 'wingDisc'
# Command to run surface evolver in ubuntu
# Refer to installationa nd usage of SE in readme 
se_path = "/home/nkumar4/Desktop/evolver_installation/src/evolver wingDisc.fe"

"""
STEP 1: Reading in the target shape data (point around which sensitivity has to be calculated)
"""
# Reading the vertices output file from a sample SE simulation output with known parameters
fe_exp = FeatureExtractor('input_data/vertices_target_SE.txt', 'input_data/log_edges.xlsx')
# Extracting the efd coefficients
dummy11, coeffs_exp_basal, dummy12, dummy21, coeffs_exp_apical, dummy22 = fe_exp.tissue_efd_coeff(num_harmonic_efd)
# Obtaining normalized x and y coordinates for the apical surface of the tissue
xt_exp_apical, yt_exp_apical = spatial_efd.inverse_transform(coeffs_exp_apical, harmonic=num_harmonic_efd)
exp_data_apical = np.zeros((len(xt_exp_apical),2))
exp_data_apical[:,0] = xt_exp_apical
exp_data_apical[:,1] = yt_exp_apical
# Obtaining normalized x and y coordinates for the basal surface of the tissue
xt_exp_basal, yt_exp_basal = spatial_efd.inverse_transform(coeffs_exp_basal, harmonic=num_harmonic_efd)
exp_data_basal = np.zeros((len(xt_exp_basal),2))
exp_data_basal[:,0] = xt_exp_basal
exp_data_basal[:,1] = yt_exp_basal

"""
STEP 2: Sensitivity analysis
"""
# initializing the error array for storing the frechet errors
error_target_sampled_apical = []
error_target_sampled_basal = []
# Initializing the array for storing the parameter values
param_sampled = np.zeros((n_param_model*2,35))
curvature_basal_master = np.zeros((n_param_model*2,129))
tissue_edge_length_master = np.zeros((n_param_model*2,390))

# Counter for iterations
k = 0
for i in range(n_param_model):	
	""" 
	Step 2A
	"""
	# Parameter values for the target shape
	paraminputs = paraminputs_stable.copy()
	# Calculating the upper and lower bound for ith parameter of interest by going up and down 70% of the set value
	param_upper_bound = paraminputs[param_index_model[i]] + 0.7*paraminputs[param_index_model[i]]
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
		# Writing geometry file
		GeometryWriter(paraminputs, param_pressure, se_filename)
		# Running surface evolver simulations
		os.system(se_path)
		# Calling in the featureExtractor class to extract geometrical features of the output data
		fe = FeatureExtractor('vertices.txt', 'input_data/log_edges.xlsx')
		# Extracting EFD coefficients and the xy coordinates of the normalized apical and basal surface of tissue
		dummy11, efd_coeff_sampled_basal, dummy12, dummy21, efd_coeff_sampled_apical, dummy22 = fe.tissue_efd_coeff(num_harmonic_efd)
		xt_sampled_apical, yt_sampled_apical = spatial_efd.inverse_transform(efd_coeff_sampled_apical, harmonic=num_harmonic_efd)
		xt_sampled_basal, yt_sampled_basal = spatial_efd.inverse_transform(efd_coeff_sampled_basal, harmonic=num_harmonic_efd)
		# Calculating the curvature of teh basal epitehlia (Outer surface)
		curvature_basal_sampled = fe.tissue_local_curvature()
		curvature_basal_sampled_reshaped = np.reshape(curvature_basal_sampled, (1, 129))
		curvature_basal_master[k,:] = curvature_basal_sampled_reshaped
		# Calculating the length of all teh edges in surface evolver model
		tissue_edge_length_sampled = fe.edge_length()
		tissue_edge_length_sampled_reshaped = np.reshape(tissue_edge_length_sampled, (1, 390))
		tissue_edge_length_master[k,:] = tissue_edge_length_sampled_reshaped
		# MOving the counter forward
		k = k + 1
		
		"""
		STEP 3C: Plotting the sampled and target shape
		"""
		# Defining filename for plot showing overlap between the sampled shape and the target shape
		filename_shape_plot_apical = str(i) + str(j) + "apical_sapled_target_xy_plot.svg"
		filename_shape_plot_basal = str(i) + str(j) + "basal_sapled_target_xy_plot.svg"
		# Plotting target data
		plt.plot(xt_exp_apical,yt_exp_apical,'black', label='Target')
		# Plotting sampled data
		plt.plot(xt_sampled_apical, yt_sampled_apical,'blue', label='Sampled')
		# Labeling axes
		plt.xlabel("x [nondimensional]")
		plt.ylabel("y [nondimensional]")
		# Plotting legends
		plt.legend()
		plt.savefig("contour_evolution_plots/" + filename_shape_plot_apical)
		plt.close()
		# plot
		plt.plot(xt_exp_basal,yt_exp_basal,'black', label='Target')
		plt.plot(xt_sampled_basal, yt_sampled_basal,'blue', label='Sampled')
		plt.xlabel("x [nondimensional]")
		plt.ylabel("y [nondimensional]")
		plt.legend()
		plt.savefig("contour_evolution_plots/" + filename_shape_plot_basal)
		plt.close()
		
		"""
		STEP 3D: Caculating frechet error
		"""
		sampled_data_apical = np.zeros((len(xt_sampled_apical),2))
		sampled_data_basal = np.zeros((len(xt_sampled_basal),2))
		sampled_data_apical[:,0] = xt_sampled_apical
		sampled_data_apical[:,1] = yt_sampled_apical
		sampled_data_basal[:,0] = xt_sampled_basal
		sampled_data_basal[:,1] = yt_sampled_basal
		# Caluclating frechet distance to emasure erros of apical and basal surfaces
		error_target_sampled_step_apical = similaritymeasures.frechet_dist(exp_data_apical,sampled_data_apical)
		error_target_sampled_step_basal = similaritymeasures.frechet_dist(exp_data_basal,sampled_data_basal)
		# Appending the data to the master array
		error_target_sampled_apical.append(error_target_sampled_step_apical)
		error_target_sampled_basal.append(error_target_sampled_step_basal)
		
		# Saving vertices 
		command_save_vertices = "cp vertices.txt vertices_" + str(i) + "_" + str(j) + ".txt"
		os.system(command_save_vertices)
		# Deleting unnecesssary surface evolver files
		os.system("rm vertices.txt")
		os.system("rm energylog.txt")
		os.system("rm specificenergylog.txt")
		
		# Removing unnecessary variables
		del xt_sampled_apical
		del yt_sampled_apical
		del xt_sampled_basal
		del yt_sampled_basal
		del fe
		gc.collect()
		

"""Saving important arrays for further analysis
"""	
np.save('output_data_files/error_target_sampled_apical.npy',error_target_sampled_apical)
np.save('output_data_files/error_target_sampled_basal.npy',error_target_sampled_basal)
np.save('output_data_files/curvature_basal_master.npy',curvature_basal_master)
np.save('output_data_files/tissue_edge_length_master.npy',tissue_edge_length_master)
np.save('output_data_files/param_sampled.npy',param_sampled)

