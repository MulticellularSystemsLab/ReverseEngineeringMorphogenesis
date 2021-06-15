# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 19:51:59 2021

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
from scipy.stats import norm


class acqisitionFunctions:
	""" A class containing functions that can be used to evaluate acquisition functions.
		Following acquisition functions can be calculated:
			1) Expected Improvement
	"""
	
	
	def __init__(self, Xsampled, x_test, y_test):
		""" Initialize
		1. Xsampled: Points at wich acquisition function needs to be calculated (n_sampled x num_parameters)
		2. x_test: Input test data for which gpr mmodel is tested
		3. y_test: True predictions for input x
		"""
		self.Xsampled = Xsampled
		self.x_test = x_test
		self.y_test = y_test
		
	def expected_improvement(self, model_gpr, likelihood_gpr, exploration_parameter):
		""" Input:
			1. model_gpr:
			2. likelihood_gpr
			3. exploration_parameter
			
			Operation: 
				Calcualtes the expected improvent at the points sampled in the parameter space
				Formula:
					1. x_optimim = argmax(y_test): calculate y_test(x_optimum) 
					2. improvement = (mu_gpr_prediction(Xsampled) - y_test(x_optimum) - exploration parameter)
					3. Z = improvement / sigma_gpr_prediction(Xsampled)
					4. expected_improvement = improvement*cdf(Z) + sigma_gpr_prediction(Xsampled)*pdf(Z)
					
				Reference:
					http://ash-aldujaili.github.io/blog/2018/02/01/ei/
					http://krasserm.github.io/2018/03/21/bayesian-optimization/
					
			Output:
				Expected improvement
		"""
		# Get into evaluation (predictive posterior) mode
		model_gpr.eval()
		likelihood_gpr.eval()
		# Convert the sampled numpy array to a tensor format for predictions
		Xsampled_torch = torch.from_numpy(self.Xsampled)
		# Make predictions by feeding model through likelihood
		with torch.no_grad(), gpytorch.settings.fast_pred_var():
			# mean and variance prediction
			observed_prediction = likelihood_gpr(model_gpr(Xsampled_torch))
			model_predictions_mean = observed_prediction.mean.numpy()
			model_predictions_variance = observed_prediction.variance.numpy()
		
		# x_optimim = argmax(y_test): calculate y_test(x_optimum)
		f_optimum = np.max(self.y_test)
		x_optimum = np.argmax(self.y_test)
		
		with np.errstate(divide='warn'):
			# improvement = (mu_gpr_prediction(Xsampled) - y_test(x_optimum) - exploration parameter)
			improvement_var = model_predictions_mean - f_optimum - exploration_parameter
			# Z = improvement / sigma_gpr_prediction(Xsampled)
			z_var = improvement_var / model_predictions_variance
			# improvement*cdf(Z)
			exploitation_term = improvement_var*norm.cdf(z_var)
			# sigma_gpr_prediction(Xsampled)*pdf(Z)
			exploration_term = model_predictions_variance*norm.pdf(z_var)
			# expected_improvement
			ei = exploitation_term + exploration_term
			# Setting expected improvement as 0where sigma_prediction is 0
			ei[model_predictions_variance == 0] = 0
		
		return ei
		
		