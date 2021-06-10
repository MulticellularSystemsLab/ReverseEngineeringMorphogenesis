# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 23:49:45 2021

@author: Nilay
"""

class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(ExactGPModel, self).__init__(train_x, train_y, likelihood
	   self.mean_module = gpytorch.means.ConstantMean()
	   self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
	   
   def forward(self, x):
	   mean_x = self.mean_module(x)
	   covar_x = self.covar_module(x)
	   return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
   
   