# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 22:14:09 2021

@author: Nilay
"""

import math
import torch
import gpytorch
import pandas as pd
import matplotlib.pyplot as plt
import math 
import numpy as np
import os
import os.path

class gaussianProcessRegression:
	

	def __init__(self, train_x, train_y_all_pc, test_x, test_y_all_pc):
		"""
		Initializion:
				train_x: Input training data for the GPR (numpy array, [split_size, num_parameters])
				train_y_all_pc: Output training data for the for the GPR model (numpy array, [split_size, num_pc])
				test_x: Input testing data for the GPR (numpy array, [num_samples-split_size, num_parameters])
				test_y_all_pc: Output testing data for the for the GPR model (numpy array, [num_samples-split_size, num_pc])
		"""
		self.train_x = train_x
		self.train_y_all_pc = train_y_all_pc
		self.test_x = test_x
		self.test_y_all_pc = test_y_all_pc
		
	
	
	def GP_model_definition(self, ExactGPModel, pc_index, model_initialize_check):
		"""
		Arguements:
			ExactGPModel: A class containing settings (kernel definition and priors) for training a many-one GP model
			pc_index: The principal component for which the model has to be trained and tested
			model_initialize_check: If 1 kernel lengthscale hyperparameters are initializes in all feature dimensions separately
			
		Operations:
			The portion of code is used to define and initialize a GP Regrression model 
			
		Output:
			model: Non trained GPR model
			likelihood: Non trained Likelihood of the model
			train_x_t, ...., test_y: Tensor form input and oututs for the GPR model
		"""
		# Converting the numpy arrays to torch tensor format
		train_x_t = torch.from_numpy(self.train_x)
		train_y_t = torch.from_numpy(self.train_y_all_pc[:,pc_index])
		test_x_t = torch.from_numpy(self.test_x)
		test_y = self.test_y_all_pc[:,pc_index]
		
		# initialize likelihood and model		
		likelihood = gpytorch.likelihoods.GaussianLikelihood()
		# Defining models for GPR
		model = ExactGPModel(train_x_t, train_y_t, likelihood)
		# Defining model hperparameters for initialization
		hypers = {
				'covar_module.base_kernel.lengthscale': torch.tensor([[1, 1, 1, 1, 1, 1, 1]]),
				}
		if model_initialize_check == 1:
			model.initialize(**hypers)
			print("Kernel lengthscale has been initialized")
		
		return model, likelihood, train_x_t, train_y_t, test_x_t, test_y
	
	def GP_training(self, train_x_t, train_y_t, model, likelihood, optimizer_select, training_iter):
		""" 
		Arguements:
			train_x_t:
			train_y_t:
			model:
			likelihood:
			optimizer_select: Select 1 or 2 to use the following optimizer type
				1. Adam Optimizer
				2. LBFGS optimizer
			training_iter:
				
		Operations:
			This section of the code trains the GP model based on the choice of optimzer. The section also compute 
			the evolution of lengthscale parameters over the training period
			
		Returns:
			model: Trained GP model
			likelihood: Trained likelihood for the GP model 
		"""
		lengthscale_hyper = np.zeros((training_iter, 7))
		model.train()
		likelihood.train()
		
		if optimizer_select == 1:
			print("Using Adam Optimizer")
			# Use the adam optimizer
			optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
			# "Loss" for GPs - the marginal log likelihood
			mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
			for i in range(training_iter):
				# Zero gradients from previous iteration
				optimizer.zero_grad()
				# Output from model
				output = model(train_x_t)
				# Calc loss and backprop gradients
				# Calc loss and backprop gradient
				loss = -mll(output, train_y_t)
				loss.backward()
				print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
				optimizer.step()
				
		elif optimizer_select == 2:
			print("Using LBFGS Optimizer")
			# Importing in the torch version of LBFGS
			from LBFGS import FullBatchLBFGS
			# Use the LBFGS algoriuthm for optimization
			optimizer =  FullBatchLBFGS(model.parameters())
			# "Loss" for GPs - the marginal log likelihood
			mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
			
			#  Reference for using LBFGS: https://github.com/hjmshi/PyTorch-LBFGS/blob/master/examples/Gaussian_Processes/gp_regression.py
			def closure():
				# zero gradients from previous iterations
				optimizer.zero_grad()
				# Output from model
				output = model(train_x_t)
				# Cslculating loss
				loss = -mll(output, train_y_t)
				return loss
			loss = closure()
			# Back propagation
			loss.backward()
			
			# Optimizing model hyperparameters
			for i in range(training_iter):
				options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
				loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
				lengthscale_hyper[i,:] = model.covar_module.base_kernel.lengthscale.detach().numpy()
				print('Iter %d/%d - Loss: %.3f ' % (i + 1, training_iter, loss.item()))
		# Returning trained model and likelihood		
		return model, likelihood, lengthscale_hyper
	
	def GP_plotting(self, test_x_t, test_y, model, likelihood, fileNameID, pc_index):
		"""
		Arguements:
			test_x_t:
			test_y:
			model:
			likelihood:
			filenameID:
			pc_index:
				
		Operations:
			
		Returns:
			Sves the plot where the predicted output is plotted against test data.
		"""
		
		model.eval()
		likelihood.eval()
		# # Make predictions by feeding model through likelihood
		with torch.no_grad(), gpytorch.settings.fast_pred_var():
			observed_pred = likelihood(model(test_x_t))
		with torch.no_grad():
			# Calculating upper and lower bounds of model predictions
			lower, upper = observed_pred.confidence_region()
			# converting upper and lower bound prediction sto numpy array
			lower_numpy = lower.numpy()
			upper_numpy = upper.numpy()
			# Claculating mean prediction
			output_model_predictions = observed_pred.mean.numpy()
			# fetching actual output data
			original_output = test_y
			
		filename = str(fileNameID) + str(pc_index) + ".png"
		# Calculating total error in predictions
		error_prediction = np.subtract(upper_numpy, lower_numpy)
		# Discretizing coordinate system for updating the parietal_plots
		x_par = np.linspace(np.amin(original_output),np.amax(original_output), num = 100)
		# Plotting the parietal line y = x
		plt.plot(x_par, x_par)
		# Plotting the output predictions against known output value
		plt.plot(original_output, output_model_predictions, 'o', color='black')
		# Plotting the errorbars
		plt.errorbar(original_output, output_model_predictions,
			   yerr = error_prediction, lolims = lower_numpy, uplims = upper_numpy, linestyle = "None")
		# Labelling axes
		plt.xlabel("True PC-" + str(pc_index+1),color="red")
		plt.ylabel("Predicted PC-" + str(pc_index+1),color="red")
		plt.ylim([-20, 20])
		# Saving the figure
		plt.savefig("gpr_model_accuracy_plots/" + filename)
		#plt.show()
		plt.close()
		
		
		
		
		
		
