# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 23:35:16 2021

@author: Nilay
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
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from smt.sampling_methods import LHS
# Importing helper libraries for bayesian optimization
from data_preprocessing_class import dataPreprocessing
from gaussian_process_regression_class import gaussianProcessRegression
from acquisition_functions_class import acqisitionFunctions
from geometry_writer import geometryWriter
from feature_extractor_4 import FeatureExtractor

"""
STEP 0:
Load the input and output data gennerated by SE model for building a GPR model
"""
# Checking if data exists
doesDataFileExist = os.path.isfile("master_feature_output.npy")
# Loading datafiles if they exist
# Else fetching and preparing data from signac workspace
if doesDataFileExist == True:
	master_parameter_input_n = np.load('master_parameter_input_n.npy', )
	master_feature_output = np.load('master_feature_output.npy', )
	
"""
STEP 2a: Data preprocessing
1. Selects the parameters sampled in LHS from total 35 parameters of the SE model
2. PCA on the out shape features.
"""
data_efd_mean = np.mean(master_feature_output,axis = 0)
data_efd_variance = np.std(master_feature_output,axis = 0)
# Loading in the data processing class
dataPreprocess  = dataPreprocessing(master_parameter_input_n, master_feature_output, 133)
# Converting the input parameters to logscale
master_parameter_input_log = dataPreprocess.inputLogTransform()
# Selecting the parameters that were sampled in the latin hypercube sampling
num_parameters_LHS = 7
LHS_parameter_index = [1, 4, 7, 17, 18, 19, 33]
# Calling in the function to separate out the desired parameters
data_x = dataPreprocess.inputParameterSelection(num_parameters_LHS, LHS_parameter_index, master_parameter_input_log)
# PCA to reduce dimensionality of the output data
total_variance_explained, principalComponents, weights = dataPreprocess.pcaEfdFeatures(8)
# Storing mean and standard deviation of input training data for later use
data_x_mean = np.mean(data_x, axis=0)
data_x_variance = np.std(data_x, axis=0)
# Normalizing data
data_x = StandardScaler().fit_transform(data_x)
# Selecting PC1 as our 1st output to the multidimensional input
data_y = principalComponents[:,0]

"""
Step 2b: Extracting EFD coefficients from the target experimental image data
a) Extracts EFD coeeficients
b) Trandforms to PC space
c) Calculates PC1 or y_target for calculation of the acquisition function
"""
geometry_data_target_disc = 'vertices_target.txt'
if type(geometry_data) is str:
	if os.stat(geometry_data).st_size != 0:
		a1 = []
		a2 = []
		with open(geometry_data) as f:
			for line in f:
				data = line.split()
				a1.append(float(data[0]))
				a2.append(float(data[1]))
				
	else:
		a1 = 0
		a2 = 0
		
vpos_x_exp = a1
vpos_y_exp = a2

coeff_exp = spatial_efd.CalculateEFD(vpos_x_exp, vpos_y_exp, 20)
# Normalizing the coefficients against rotation and size
coeff_exp, rotation = spatial_efd.normalize_efd(coeff_exp, size_invariant=True)
# Reverse EFD for plotting the normalized tissue shape
xt, yt = spatial_efd.inverse_transform(coeff_exp, harmonic=20)
efd_coeff_exp_reshaped = np.reshape(coeff_exp, (80,))

efd_coeff_exp_normalized = (np.add(np.multiply(efd_coeff_exp_reshaped,data_efd_variance), data_efd_mean)) 
efd_coeff_exp_normalized = np.reshape(efd_coeff_exp_normalized, (80,1))
# Multiplying EFD coefficients by already obtained weight of pc
efd_coeff_exp_normalized_pc = np.matmul(weights,efd_coeff_exp_normalized)
# Reshaping array for appending to the original data array
y_exp = np.reshape(efd_coeff_exp_normalized_pc, (1,8))
y_target = efd_coeff_exp_normalized_pc[0]
		
	 
"""
Step 3: class ExactGP Model 
        Needs to be executed to execute the 
		  gaussianProcessRegression class 
"""
class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
		
	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
# Calling in the gpr class
gpr  = gaussianProcessRegression(data_x, data_y)
# Splitting up the training and test data
train_x, train_y, test_x, test_y = gpr.split_data(110, 133)

"""
Step 5: Executing the loop for bayesian optimization

"""
maxIter = 50

for i in range(maxIter):
	""" Step 5a: Training the GP model 
	"""
	# Getting the trained model and likelihood using the training data
	model, likelihood = gpr.GP_Regressor(train_x, train_y, test_x, test_y, 1000, i+1, ExactGPModel)
	
	""" Step 5b: Estimating acquisition function
			Sample random points in space
			Use the gpr model to estimate expected improvement
			find optimum x
			Transform to parameter space
	"""
	# Method I: latine hypercube sampling
	xlimits = np.array([[-2, 2],[-2, 2],[-2, 2],[-2, 2],[-2, 2],[-2, 2],[-2, 2]])
	sampling = LHS(xlimits = xlimits)
	# Defining numvber of samples
	num_samples = 1000000
	# Implementing latin hypercube sampling
	x = sampling(num_samples)
	
	# Method II: Random sampling
	num_samples = 1000000
	x = np.random.rand(num_samples, 7)
	
	# Calling in the acquisition function class
	af = acqisitionFunctions(x, test_x, test_y)
	# Calculating the xpected improvement
	ei = af.expected_improvement(model, likelihood, 0.9, y_target)
	# Finding the indez that leads to maximum acquisition function
	x_sampled_index = np.argmax(ei)
	# Assessing the new sampled value
	x_sampled_logscale_standardized = x[x_sampled_index,:]
	# Converting x sampled into parameter space
	# Multiplying by standard deviation and adding the mean pf data
	x_sampled = 10**(np.add(np.multiply(x_sampled_logscale_standardized,data_x_variance), data_x_mean))
	
	"""Step 5c: Run surface evolver simulations
	"""
	# Initializaib=ng the surface evolver parameters
	paraminputs = [0,0.0001,0,0,0,0,0,0.001,0,0,0, 0.1,0.1,10,0.1,0.1,0.1,0.1,10,0.0001,0.001,0.001, 1,1,0.6,0.6,0.6,0.6,0.2,0.1,3,0.6,1.8, 0.001,0.001]
	# Repalcaing the parameters with newly sampled values
	paraminputs[1] = x_sampled[0,]
	# tension cuboidal basal
	paraminputs[4] = x_sampled[1,]
	# tension columnar basal
	paraminputs[7] = x_sampled[2,]
	# k columnar apical
	paraminputs[17] = x_sampled[3,]
	# k columnar basal
	paraminputs[18] = x_sampled[4,]
	# k columnar lateral
	paraminputs[19] = x_sampled[5,]
	# K_ECM
	paraminputs[33] = x_sampled[6,]
	# Defining the set system pressure
	param_pressure = 0.001
	# Writing geometry file
	geometryWriter(paraminputs, param_pressure, 'wingDisc')
	# Running surface evolver simulations
	# Can be replaced using python subprocess using 
	os.system("/home/nkumar4/Desktop/evolver_installation/src/evolver wingDisc.fe")
	
	""" Step 5d : Extracting EFD features from outsimulation data
	"""
	fe = FeatureExtractor('vertices.txt', 'log_edges.xlsx')
	efd_coeff_sampled = fe.tissue_efd_coeff(20)
	efd_coeff_sampled_reshaped = np.reshape(efd_coeff_sampled, (80,))
	
	"""Step 5e: Converting EFD to PC space
	"""
	# normalizing efd coefficients with existing data mean and variance
	efd_coeff_sampled_normalized = (np.divide(np.subtract(efd_coeff_sampled_reshaped,data_efd_mean), data_efd_variance))
	efd_coeff_sampled_normalized = np.reshape(efd_coeff_sampled_normalized, (80,1))
	# Multiplying EFD coefficients by already obtained weight of pc
	efd_coeff_sampled_normalized_pc = np.matmul(weights,efd_coeff_sampled_normalized)
	# Reshaping array for appending to the original data array
	y_sampled = np.reshape(efd_coeff_sampled_normalized_pc, (1,8))
	
	""" Step 5f: Updating training data
	"""
	data_x = np.vstack((data_x, np.reshape(x_sampled,(1,7))))
	
	""" Step 5g: Removing files for next iteration
	"""
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	


 





print(np.shape(efd_coeff_sampled_normalized_pc))

"""STEP 5: UPDATING TRAINING DATA"""
#data_y = np.vstack((data_y, y_sampled[0,0]))

print(np.shape(data_x))








