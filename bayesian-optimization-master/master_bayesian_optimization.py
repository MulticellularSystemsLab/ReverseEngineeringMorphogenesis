# -*- coding: utf-8 -*-
"""
Main code for carrying out Bayesian optimization (BO) to estimate Surface Evolver model parametsrs from
wing imaginal disc tissue cross section data

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
User deribed inputs
"""
# Number of parameters that need to be estimated during BO 
num_parameters_LHS = 7
# List containing parameter indices that need to be estimated using the BO framework
LHS_parameter_index = [17, 18, 19, 28, 29, 30, 33]
# Path to the target shape data for which teh parameters need to be estimated
# It should be a .txt file containing x y coordinates of points belonging to wing imaginal disc external surface 
geometry_data = 'input_data/case1_pzornai_ctrl.txt'
# geometry_type = 1: data belongs to an experimental crosss ection
# geometry_type = 2: synthetic data from a surface evolver output
geometry_data_type = 1
# Total number of parameter sets samoled during calculation of acquisition function
num_samples_af = 100000
# Total number of iteration for the BO process
n_iterations = 3
# Number of samples from the total samples taht will constitute the training data
split_size = 149
# Total number of data points that are used fro tarining the GP model
num_samples = 150
# A sample parameter set where all the parameters except the ones varoed are kept similar to one during the LHS smapling for generating training data.
paraminputs_stable = [0,0.0001,0,0,0,0,0,0.001,0,0,0, 0.1,0.1,10,0.1,0.1,0.1,0.1,10,0.0001,0.001,0.001, 1,1,0.6,0.6,0.6,0.6,0.2,0.1,3,0.6,1.8, 0.001,0.001]
# A parameter to define pressure of the system
param_pressure = 0.001
# Total number of harmonics need for defining elliptic fourier decriptor coefficients for representing the pouch external surface
num_harmonics_efd = 20
# A parameter to define the tradeoff between exploration and exploitation during BO
exploration_param_val = 0.05
# Total number of iterations for training the GP model
num_iteration_gpr = 5000
# Selecting the type of optimizer used for training of GP model 
# 1: Adam Optimizer 2: LBFGS 
optimizer_type = 1
# Name of the file generated as a surface evolver output
# 	NOTE: Make sure to change the next line if changineg this
se_filename = 'wingDisc'
# Path containing the surface evolver installation and se filename to run it
se_path = "/home/nkumar4/Desktop/evolver_installation/src/evolver wingDisc.fe"


"""
STEP 1:
Load the input and output data gennerated by SE model for building a GPR model.
The input data should consist of a [num_samples x 35] and the output data  contains the EFD coeffiecients
should be of size [num_samples x 4*num_harnmonics_efd]
"""
# Checking if data exists
doesDataFileExist = os.path.isfile("input_data/master_feature_output.npy")
# Loading datafiles if they exist
# Else fetching and preparing data from signac workspace
if doesDataFileExist == True:
	master_parameter_input_n = np.load('input_data/master_parameter_input_n.npy', )
	master_feature_output = np.load('input_data/master_feature_output.npy', )
	
"""
STEP 2: 
Input data preprocessing - Preparing inputs and outputs for the GPR model
1. Input data: Selects the parameters sampled in LHS from total 35 parameters of the SE model. 
   The resulting input data to the GP model should be of size [num_samples x 7]
2. Output data: Traget shape Shape for whicb parameter estimation has to be carried out is loaded first.
   Frachet distance is used to evaluate the error between the sampled parameters and target shape.
   Lastly a negative of the Frechet error is taken that constituet the output training data for the gP model
	   A. geometry_data_type == 1: target shape is derived from experimental data
	   B. geometry_data_type == 2: Target shape is generated from surface evolver (synthetic)
"""
# Loading in the data processing class
dataPreprocess  = DataPreprocessing(master_parameter_input_n, master_feature_output, num_samples)
# Converting the input parameters to logscale
master_parameter_input_log = dataPreprocess.input_log_transform()
# Calling in the function to separate out the desired parameters
data_x = dataPreprocess.input_parameter_selection(num_parameters_LHS, LHS_parameter_index, master_parameter_input_log)
# Storing mean and standard deviation of input training data for later use
data_x_mean = np.mean(data_x, axis=0)
data_x_variance = np.std(data_x, axis=0)
# Normalizing data
data_x = StandardScaler().fit_transform(data_x)
# Calculating the minimum and maximum for input parameters for the purpose of sampling the parameters using LHS later
max_data_x = np.amax(data_x, axis=0) 
min_data_x = np.amin(data_x, axis=0) 

# Reading in experimental data as a list of xy points representing the tissue lateral shape
# If data is derived from experiments
if geometry_data_type == 1:
	if type(geometry_data) is str:
		# Reading in data as two separate list of x and y coordinates 
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
	
	# Calculating EFD coefficients 
	coeff_exp = spatial_efd.CalculateEFD(vpos_x_exp, vpos_y_exp, num_harmonics_efd)
	# Normalizing the coefficients against rotation and size
	coeff_exp, rotation = spatial_efd.normalize_efd(coeff_exp, size_invariant=True)

# Surface evolver derived synthetic data	
elif geometry_data_type == 2:
	# Calling in the feature extraction class for calculation of elliptic fourier descriptrs
	fe_exp = FeatureExtractor(geometry_data, 'input_data/log_edges.xlsx')
	coeffs_exp, dummy_1, dummy_2 = fe_exp.tissue_efd_coeff(num_harmonics_efd)

# Calculating teh normalized x and y coordinates using a reverse EFD iperation 
xt_exp, yt_exp = spatial_efd.inverse_transform(coeff_exp, harmonic=num_harmonics_efd)
exp_data = np.zeros((len(xt_exp),2))
# String the x and y coordinates in a different variable named exp_data
exp_data[:,0] = xt_exp
exp_data[:,1] = yt_exp

# Calculating the frechet distance betwen the target shape and the simulation data present in the master_feature_output data
# Initializing the array to store the output data for GP mode
error_simulation_experimental_data = np.zeros(num_samples)
# Itaetaing through the number of samples
for i in range(num_samples):
    temp = master_feature_output[i,:] # Reading in the efd coefficients
    temp2 = np.reshape(temp, (num_harmonics_efd,4)) # Reshaping the EFD coefficensts for taking reverse EFD
    xt, yt = spatial_efd.inverse_transform(temp2, harmonic=num_harmonics_efd) # Perfroming reverse EFD to obtaing x y coordinates of teh basal contour
    sim_data = np.zeros((len(xt),2)) # Initializing an array to store the xy data normalzied for simulations
    sim_data[:,0] = xt
    sim_data[:,1] = yt
    error_simulation_experimental_data[i] = similaritymeasures.frechet_dist(exp_data,sim_data) # Calculating Frechet distance
    
# Taking a negativbe of the Frechet distance to genrate the input data for teh GP model
data_y = (np.reshape(error_simulation_experimental_data, (num_samples,1)))*(-1)
print(np.shape(data_y))
			 
"""
STEP 3: 
1) Define a ExactGP class containing GP model settings for the gaussianProcessRegression class
2) The input and output data is next split into training and test data
3) lastly sample a large number of points for computation of acquisition function
"""
class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		# tHE LENGTH SCALE OF THE COVARIANCE MATRIX IS KEPT DIFFERENT FOR DIFFERENT DIMENSIONS
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=7))
		# self.white_noise = gpytorch.kernels.ScaleKernel(gpytorch.kernels.WhiteNoiseKernel())
		
	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Splitting teh input data into tarining and test datasets
train_x = data_x[:split_size,:]
train_y = data_y[:split_size,:]
test_x = data_x[split_size:num_samples,:]
test_y = data_y[split_size:num_samples,:]
 
# Sampling the parameter space for clculation of acquisition function
xlimits = np.array([[min_data_x[0], max_data_x[0]],[min_data_x[1], max_data_x[1]],[min_data_x[2], max_data_x[2]],[min_data_x[3], max_data_x[3]],[min_data_x[4], max_data_x[4]],[min_data_x[5], max_data_x[5]],[min_data_x[6], max_data_x[6]]])
sampling = LHS(xlimits = xlimits)
x = sampling(num_samples_af)

"""
step 5: Start the BO loop
1. Train the GP model
2. Compute expected improvedment
3. Sample a new point
4. Run surface evolver
5. Compute the error between the surface evolver geenerated shape for teh sampled parameter and training data
6. Add the smapled parameter and error computed to the training data. (post Normalizationa nd sign conversion)
7. Redo 1 with new training data

"""
# Initializing array to stor ethe rror between the target shaoe and the shape generated by SE using teh sampled parametsr
error_target_sampled = []
# Initializng the counter for storing the iteration number
iter_counter = []
# Initializng teh array for storing teh parametsr sampled during the optimization process
param_sampled = np.zeros((n_iterations,7))

for i in range(n_iterations):
	""" 5.1 Training the GPR model """	
	# Initializng the GaussianProcessRegression class
	gpr = GaussianProcessRegression(train_x, train_y, test_x, test_y)
	# Using the method GP_model_definition to initialzie teh GP model parameters and training data
	model, likelihood, train_x_t, train_y_t, test_x_t, test_y_pc = gpr.GP_model_definition(ExactGPModel,0,1)
	# Traing the GPR model
	model, likelihood, lengthscale_hyperparameters = gpr.GP_training(train_x_t, train_y_t,model,likelihood,optimizer_type, num_iteration_gpr)
	
	""" 5.2 Calculating expected imrovement """
	# Initializng the AcqisitionFunctions class
	af = AcqisitionFunctions(x, train_x, train_y)
	# Computing expected improvenet for the sampled points
	ei, model_prediction_mean, model_prediction_variance = af.expected_improvement(model, likelihood, exploration_param_val)
	
	""" 5.3 Using EI to sample a new point """
	# Sampling a new point in teh parameter space
	x_sampled_index = np.argmax(ei)
	x_sampled_logscale_standardized = x[x_sampled_index,:]
	# Converting x sampled into parameter space
	x_sampled = np.exp(np.add(np.multiply(x_sampled_logscale_standardized,data_x_variance), data_x_mean))
	
	""" 5.4 Running surface evolver for the sampled parameters"""
	# Initializaib=ng the surface evolver parameters
	paraminputs = paraminputs_stable.copy()
	# Repalcaing the parameters with newly sampled values
	paraminputs[LHS_parameter_index[0]] = x_sampled[0,]
	# tension cuboidal basal
	paraminputs[LHS_parameter_index[1]] = x_sampled[1,]
	# tension columnar basal
	paraminputs[LHS_parameter_index[2]] = x_sampled[2,]
	# k columnar apical
	paraminputs[LHS_parameter_index[3]] = x_sampled[3,]
	# k columnar basal
	paraminputs[LHS_parameter_index[4]] = x_sampled[4,]
	# k columnar lateral
	paraminputs[LHS_parameter_index[5]] = x_sampled[5,]
	# K_ECM
	paraminputs[LHS_parameter_index[6]] = x_sampled[6,]
	# Writing geometry file
	GeometryWriter(paraminputs, param_pressure, se_filename)
	# Running surface evolver simulations
	# Can be replaced using python subprocess using 
	os.system(se_path)
	# Initializng the FeatureExtractor class usibg the output from surface evolver (vertices.txt)
	fe = FeatureExtractor('vertices.txt', 'input_data/log_edges.xlsx')
	# Extracting the EFD coefficients
	efd_coeff_sampled, dummy_3, dummy_4 = fe.tissue_efd_coeff(num_harmonics_efd)
	# Using reverse EFD to obtain the x y coordinates of the normalized external tissue surface
	xt_sampled, yt_sampled = spatial_efd.inverse_transform(efd_coeff_sampled, harmonic=num_harmonics_efd)
	# Saving the SE output data of teh samples point 
	command_save_vertices = "cp vertices.txt vertices_" + str(i) + ".txt"
	os.system(command_save_vertices)
	os.system("rm vertices.txt")
	os.system("rm energylog.txt")
	os.system("rm specificenergylog.txt")
	
	# Defining filename for plot showing overlap between the sampled shape and the target shape
	filename_shape_plot = str(i) + "_sapled_target_xy_plot.svg"
	# Plotting target data
	plt.scatter(xt_exp,yt_exp, color='black')
	# Plotting sampled data
	plt.scatter(xt_sampled, yt_sampled, color='blue')
	# Labeling axes
	plt.xlabel("x [nondimensional]")
	plt.ylabel("y [nondimensional]")
	# Plotting legends
	#plt.legend()
	plt.savefig("contour_evolution_plots/" + filename_shape_plot)
	plt.close()
	
	
	""" 5.5 Calculating error: Calculating frechet distance between the sampled shape and the experimental data"""
	sampled_data = np.zeros((300,2))
	sampled_data[:,0] = xt_sampled
	sampled_data[:,1] = yt_sampled
	# Computing teh frechet distance between the sampled and target shape
	error_target_sampled_step = similaritymeasures.frechet_dist(exp_data,sampled_data)
	# Taking negative of teh error. Will be added to the training data output
	y_sampled = np.reshape(error_target_sampled_step, (1,1))*(-1)
	
	""" 5.6 Adding to the training data """
	# Adding the sampled point using acquisition function in the training data (Note we are adding teh normalized parametsr)
	train_x = np.vstack((train_x, np.reshape(x_sampled_logscale_standardized,(1,7))))
	# Adding the -ve of Frecht distance between the target and sampeld shape
	train_y = np.vstack((train_y, y_sampled))
	
	# String teh error sampled and parameter sampled in the master data
	error_target_sampled.append(error_target_sampled_step)
	# Apeendingthe stape to teh iteration counter
	iter_counter.append(i+1)
	# Adding teh smapled parametsr to teh master data for later analysis
	param_sampled[i,:] = x_sampled
	
	# Plotting the error of the sampled points against the training data
	filename_error_iteration = str(i) + "error_evolution.png"
	error_train = data_y*-1
	error_train_reshaped = np.reshape(error_train, (num_samples,))
	error_train_sorted = np.flip(np.sort(error_train_reshaped))
	index_trained = np.linspace(1, num_samples, num_samples)
	index_sampled = np.linspace(num_samples+1,num_samples+i+1,i+1)
	plt.scatter(index_trained, error_train_sorted, color="black")
	plt.scatter(index_sampled, error_target_sampled, color="red")
	plt.ylabel("Index")
	plt.xlabel("Error between target and SE shape")
	plt.savefig("error_sampled_plots/" + filename_shape_plot)
	plt.close()
	# Removing variables generated during teh iteration to handle memory issues
	del x_sampled
	del xt_sampled
	del yt_sampled
	del fe
	del gpr
	del model
	del likelihood
	del index_trained
	del index_sampled
	gc.collect()
	
""" Saving important data for further analysis """

# Error (Frechet distance) between teh target shape and sampled shape
np.save('output_data/error_target_sampled.npy',error_target_sampled)
# Parametsr sampled during the BO process
np.save('output_data/param_sampled.npy',param_sampled)
# Updated training data for teh GP model
np.save('output_data/train_x.npy', train_x)
np.save('output_data/train_y.npy', train_y)
