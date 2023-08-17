# -*- coding: utf-8 -*-
"""
Code defines Expected Improvements as a acquisition function

Created on Mon Jun 14 19:51:59 2021
@author: Nilay Kumar
email: nkumar4@nd.edu
Multicellular Systems Engineering Lab (MSELab)
Department of Chemical and Biomolecular Engineering
Institution: University of Notre Dame
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
import gc


class AcqisitionFunctions:
	""" A class containing functions that can be used to evaluate acquisition functions.
		Following acquisition functions can be calculated:
			1) Expected Improvement
	"""
	
	
	def __init__(self, Xsampled, x_test, y_test):
		""" Initialize
		1. Xsampled: Points at which acquisition function needs to be calculated (n_sampled x num_parameters)
		2. x_test: Input test data for which gpr mmodel is tested
		3. y_test: True predictions for input x
		"""
		self.Xsampled = Xsampled
		self.x_test = x_test
		self.y_test = y_test
		
	def expected_improvement(self, model_gpr, likelihood_gpr, exploration_parameter):
		""" Arguements:
			1. model_gpr, likelihood_gpr:Trained GP regression model obejcts
			2. exploration_parameter: (Float), Used during computation of expcted improvement to control exploration and expolitation
			
			Operations: 
				Calcualtes the expected improvent at the points sampled in the parameter space
				Formula:
					1. x_optimim = argmax(y_test): calculate y_test(x_optimum) 
					2. improvement = (mu_gpr_prediction(Xsampled) - y_test(x_optimum) - exploration parameter)
					3. Z = improvement / sigma_gpr_prediction(Xsampled)
					4. expected_improvement = improvement*cdf(Z) + sigma_gpr_prediction(Xsampled)*pdf(Z)
					
				Reference:
					http://ash-aldujaili.github.io/blog/2018/02/01/ei/
					http://krasserm.github.io/2018/03/21/bayesian-optimization/
					
			Return:
				   1. Expected improvement: A numpy array of size n_sampled x 1 holding the EI value for
				      the sampled points in parameter space
				   2. model_prediction_mean: A numpy array of size n_sampled x 1 holding the mean of prediction made 
				      by the GP model
				   3. model_prediction_variance: A numpy array of size n_sampled x 1 holding the variance of prediction 
				      made by the GP model
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
		
		# Handling warnings for division by 0
		# If anywhere below the code a division by 0 occurs, a warning will be genrated 
		# but the code will keep on running....
		with np.errstate(divide='warn'):
			# improvement = (mu_gpr_prediction(Xsampled) - y_test(x_optimum) - exploration parameter)
			improvement_var = model_predictions_mean - f_optimum - exploration_parameter
			# Z = improvement / sigma_gpr_prediction(Xsampled)
			z_var = np.divide(improvement_var, model_predictions_variance)
			# improvement*cdf(Z)
			exploitation_term = np.multiply(improvement_var, norm.cdf(z_var))
			# sigma_gpr_prediction(Xsampled)*pdf(Z)
			exploration_term = np.multiply(model_predictions_variance, norm.pdf(z_var))
			# expected_improvement
			ei = exploitation_term + exploration_term
			# Setting expected improvement as 0where sigma_prediction is 0
			ei[model_predictions_variance <= 1E-6] = 0
		# function returns
		return ei, model_predictions_mean, model_predictions_variance 
	
	def expected_improvement_modified(self, model_gpr, likelihood_gpr, f_target):
		"""
		Input:
			1) model_gpr, likelihood_gpr: Trained GP regression model obejcts
			2) f_target: Target PC1 of EFD extracted from raw experimental data
		Operation
			computes best error as minimum of min--x--(f_target - model_prediction)**2
			
			Computes expected improvement across sampled poits using the formula
				E1 = E1_term_1 + EI_term_2 - EI_term3
				a) EI_term_1 = (error_best-(f_target - mean_prediction_gpr)^2)(cdf(eps2)-cdf(eps1)
				b) EI_term_2 = (2(f_target-mean_prediction_gpr)xsigma_predictive_gpr)*(pdf(eps2)-pdf(eps1))
				c) EI_term_3 = {0.5*[cdf(eps2/1.414) - cdf(eps1/1.414)] + [eps1*pdf(eps1) - eps2*pdf(eps2)])*(sigma_predictive_gpr**2)
			
			The bounds eps1 and eps2 are obtained as
				(f_target - [mean_prediction_gpr) +- (sqrt(error_best)]/sigma_predictive_gpr
				
			Output:
				Expected Improvement: Expected improvement (A numpy array of size n_sampled x 1)						    
		"""
		# Get into evaluation (predictive posterior) mode
		model_gpr.eval()
		likelihood_gpr.eval()
		# Convert the sampled numpy array to a tensor format for predictions
		Xsampled_torch = torch.from_numpy(self.Xsampled)
		with torch.no_grad(), gpytorch.settings.fast_pred_var():
			# mean and variance prediction
			observed_prediction = likelihood_gpr(model_gpr(Xsampled_torch))
			model_predictions_mean = observed_prediction.mean.numpy()
			model_predictions_variance = observed_prediction.variance.numpy()
			
		# computes best error as minimum of min--x--(f_target - model_prediction)**2
		error = (np.square(f_target - self.y_test))*(-1)
		error_best = np.max(error)
		error_best = error_best*-1
		x_optimum = np.argmax(error)
		
		with np.errstate(divide='warn'):
			# Upper and lower bounds are estimated by [mean_prediction_gpr) +- (sqrt(error_best)]/sigma_predictive_gpr
			bound_upper = np.divide((f_target - model_predictions_mean + math.sqrt(error_best)), model_predictions_variance) 
			bound_lower = np.divide((f_target - model_predictions_mean - math.sqrt(error_best)), model_predictions_variance)
			# ei_term_1_component_1 = (cdf(eps2)-cdf(eps1)
			ei_term_1_component_1 = norm.cdf(bound_upper) - norm.cdf(bound_lower)
			# ei_term_1_component_2 = (error_best-(f_target - mean_prediction_gpr)^2)
			ei_term_1_component_2 = error_best - np.square(f_target - model_predictions_mean) 
			ei_term_1 = np.multiply(ei_term_1_component_1,ei_term_1_component_2)
			
			# ei_term_2_component_1 = pdf(eps2)-pdf(eps1)
			ei_term_2_component_1 = norm.pdf(bound_upper) - norm.pdf(bound_lower)
			# ei_term_2_component_2 = s2*igma_predictive_gpr
			ei_term_2_component_2 = 2 * model_predictions_variance
			# ei_term_2_component_3 = (f_target - mean_prediction_gpr)
			ei_term_2_component_3 = f_target - model_predictions_mean
			ei_term_2 = np.multiply(np.multiply(ei_term_2_component_1, ei_term_2_component_2), ei_term_2_component_3)
			
			# ei_term_3_component_1 = [eps1*pdf(eps1) - eps2*pdf(eps2)]
			ei_term_3_component_1 = np.multiply(norm.pdf(bound_lower), bound_lower) - np.multiply(norm.pdf(bound_upper), bound_upper)
			# ei_term_3_component_2 = [cdf(eps2/1.414) - cdf(eps1/1.414)]
			ei_term_3_component_2 = norm.cdf(bound_upper / math.sqrt(2)) - norm.cdf(bound_lower / math.sqrt(2))
			ei_term_3_component_3 = ei_term_3_component_1 + ei_term_3_component_2 / 2
			# Taking square of model_predictions_variance
			ei_term_3_component_4 = np.multiply(model_predictions_variance, model_predictions_variance)
			ei_term_3 = np.multiply(ei_term_3_component_3, ei_term_3_component_4)
			
			# EI = EI_term1 + EI_term_2 + EI_term_3
			ei = ei_term_1 + ei_term_2 - ei_term_3
			
			del ei_term_1_component_1
			del ei_term_1_component_2
			del ei_term_2_component_1
			del ei_term_2_component_2
			del ei_term_2_component_3
			del ei_term_3_component_1
			del ei_term_3_component_2
			del ei_term_3_component_3
			del ei_term_3_component_4
			del ei_term_1
			del ei_term_2
			del ei_term_3
			del bound_upper
			del bound_lower
			del error
			del Xsampled_torch
			gc.collect()
			
			 
		
		return ei		
		
		
		
		
		
		
		
		
		
			
			
		
		
		
