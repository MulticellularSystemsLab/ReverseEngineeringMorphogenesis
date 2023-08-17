# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 09:42:27 2021
@author: Nilay Kumar
email: nkumar4@nd.edu
Multicellular Systems Engineering Lab (MSELab)
Department of Chemical and Biomolecular Engineering
Institution: University of Notre Dame
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math 
import numpy as np
import os
import os.path


class DataPreprocessing:
	
	
	def __init__(self, master_parameter_input_n, master_feature_output, num_samples):
		""" Initialization:
			a) master_parameter_input_n: A n_samples x num_parameters numpy array containing the parametsrs for teh sampled points
			b) num_samples: Total number of data points
			c) master_feature_output: Surface Evolver is run for each parameter set and output is the frechet error of the
	         basal surface rom the target shape 
		"""
		self.master_parameter_input_n = master_parameter_input_n
		self.master_feature_output = master_feature_output
		self.num_samples = num_samples
		
		
	def input_log_transform(self):
		""" 
		Arguements:
			 master_parameter_input: A n_samples x num_parameters numpy array containing the parametsrs for teh sampled points
		Operations:
			Transforms the input parameter space into a log space
		Returns:
			Log transformed parameters (n_samplesx num_parameters)
		 
		"""
		master_parameter_input = np.log(self.master_parameter_input_n)
		return master_parameter_input
	
	def pca_efd_features(self, num_pca_components):
		"""
		Arguements:
			A) master_feature_output (EFD shape features)
			B) num_pca_components: Total number of principal component
	   Operations:
		   The function takes in the EFD features and carries out a principal componet analysis 
		   to reduce the dmensionality of the problem.
	   Returns:
		   A) total_variance_explained: Total variance captured in the transformed space. Useful for 
			identifying the number of principal components.
			B) principal_compnents: The transformed variables. Used in training the GPR model.
			C) weigts: Will be used to transform back the PCs into original the feature space.
		"""
		# Storing the EFDs in a dummy variable x
		x = self.master_feature_output
		# Scaling the EFD data
		x = StandardScaler().fit_transform(x)
		# PCA operation carried out for the desired number of components
		pca = PCA(n_components=num_pca_components)
		# Calculating the principal components for the data
		principalComponents = pca.fit_transform(x)
		# Calculating Weights
		weights = pca.components_
		# Sum of variance captured by each PC
		total_variance_explained = sum(pca.explained_variance_ratio_)
		# Weights of af
		weight_af = pca.explained_variance_ratio_ / sum(pca.explained_variance_ratio_)
		
		return total_variance_explained, principalComponents, weights, weight_af
	
	def input_parameter_selection(self, num_parameters_LHS, LHS_parameter_index, master_parameter_input_log):
		"""
		Arguements:
			A) num_parameters_LHS: number of parameters sampled in the Latin Hypercube Sampling (LHS)
			B) LHS_parameter_index: Index of the sampled parameters in the master_input_parameter array
			C) master_parameter_input_log: log transformed parameter space
		Operation:
			The function selects out the genarted para,eters from all the parameters 
			used in the surface evolver model.
		Returns:
			A) train_x_numpy: Input data contaning only the parameters sampled is the LHS
				 
		"""
		# Initializing the output array
		train_x_numpy = np.zeros((self.num_samples, num_parameters_LHS))
		# Going through each parameter index for the LHS sampled parameters
		for i in range(num_parameters_LHS):
			# Storing the para,eter value for all the data points sampled
			train_x_numpy[:,i] = master_parameter_input_log[:,LHS_parameter_index[i]]
		
		# returning the output variable
		return train_x_numpy