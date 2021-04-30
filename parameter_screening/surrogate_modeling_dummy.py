#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 18:03:32 2021

@author: nkumar4
"""

import pandas as pd
import matplotlib.pyplot as plt
import spatial_efd
import math 
import signac
import numpy as np

#project = signac.get_project()
#master_edge_length = np.zeros(390,)

class signacDataAnalysis:
	
	def _init_(self):
		self.project = signac.get_project()
		
	def efd_shape_analysis(self):
		for job in self.project:
			input_param = job.statepoint()["parameter_model"]
			input_param = np.array(input_param)
			efd_coeff = job.document.get("e_f_d")
			efd_coeff = np.array(efd_coeff)
			xt, yt = spatial_efd.inverse_transform(efd_coeff, harmonic=20)
			plt.plot(xt,yt,label=str(input_param[33]))
			plt.axes().set_aspect('equal', 'datalim')
		plt.legend(loc='upper left',ncol = 2, prop={'size': 6})   	
		plt.show()
			

for job in project:
    input_param = job.statepoint()["parameter_model"]
    input_param = np.array(input_param)
    efd_coeff = job.document.get("e_f_d")
    efd_coeff = np.array(efd_coeff)
    print(np.shape(efd_coeff))
    xt, yt = spatial_efd.inverse_transform(efd_coeff, harmonic=20)
    plt.plot(xt,yt,label=str(input_param[33]))
    plt.axes().set_aspect('equal', 'datalim')
#    
#    
#plt.legend(loc='upper left',ncol = 2, prop={'size': 6})      
#plt.show()



#for job in project:
#    input_param = job.statepoint()["parameter_model"]
#    input_param = np.array(input_param)
#    efd_coeff = job.document.get("e_f_d")
#    efd_coeff = np.array(efd_coeff)
#    print(np.shape(efd_coeff))
#    xt, yt = spatial_efd.inverse_transform(efd_coeff, harmonic=20)
#    plt.plot(xt,yt,label=str(input_param[33]))
#    plt.axes().set_aspect('equal', 'datalim')
#    
#    
#plt.legend(loc='upper left',ncol = 2, prop={'size': 6})      
#plt.show()


#for job in project:
#    input_param = job.statepoint()["parameter_model"]
#    input_param = np.array(input_param)
#    tissue_local_height = job.document.get("length")
#    tissue_local_height = np.array(tissue_local_height)
#    plt.plot(list(range(0, 390)),tissue_local_height,label=str(input_param[33]))
#    
#plt.xlabel("Edge id")
#plt.ylabel("Local tissue height")    
#plt.legend(loc='upper left',ncol = 5, prop={'size': 6})    
#plt.show()


#for job in project:
#    input_param = job.statepoint()["parameter_model"]
#    input_param = np.array(input_param)
#    curvature_local = job.document.get("curvature")
#    curvature_local = np.array(curvature_local)
#    plt.plot(list(range(0, 129)),curvature_local,label=str(input_param[33]))
#
#plt.xlabel("Edge id")
#plt.ylabel("Local curvature (1 / length)")    
#plt.legend(loc='upper center',ncol = 5, prop={'size': 6})    
#plt.show()



