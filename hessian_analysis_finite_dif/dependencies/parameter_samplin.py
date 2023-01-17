"""
Created on Thu Jul 21 09:46:55 2022

@author: Nilay
"""
"""
sampleParameters(paraminputsMaster,param_index_model,var1,var2, termid)

A) param_index_model: Array containing the indexes of parameters for which hessian analysis is done
B) var1 - Index of parameter 1 in the array above to be varied
C) var2 - Index of parameter 2 in the array above to be varied
D) termid:
	0 = Both param increased by 70%
	1 = Both param decreased by 70%
	2 = param 1 increased by 70% and param 2  decreased by 70%
	3 = param 1 decreased by 70% and param 2  increased by 70%
"""


def sampleParameters(paraminputsMaster,param_index_model,var1,var2, termid):
	
	if termid == 0:
		param_sim = paraminputsMaster
		param_sim[param_index_model[var1]] = param_sim[param_index_model[var1]] + 0.7*param_sim[param_index_model[var1]];
		param_sim[param_index_model[var2]] = param_sim[param_index_model[var2]] + 0.7*param_sim[param_index_model[var2]];
		
	elif termid == 1:
		param_sim = paraminputsMaster
		param_sim[param_index_model[var1]] = param_sim[param_index_model[var1]] - 0.7*param_sim[param_index_model[var1]];
		param_sim[param_index_model[var2]] = param_sim[param_index_model[var2]] - 0.7*param_sim[param_index_model[var2]];
		
	elif termid == 2:
		param_sim = paraminputsMaster
		param_sim[param_index_model[var1]] = param_sim[param_index_model[var1]] + 0.7*param_sim[param_index_model[var1]];
		param_sim[param_index_model[var2]] = param_sim[param_index_model[var2]] - 0.7*param_sim[param_index_model[var2]];
		
	elif termid == 3:
		param_sim = paraminputsMaster
		param_sim[param_index_model[var1]] = param_sim[param_index_model[var1]] - 0.7*param_sim[param_index_model[var1]];
		param_sim[param_index_model[var2]] = param_sim[param_index_model[var2]] + 0.7*param_sim[param_index_model[var2]];
		
	return param_sim
		
		
		
		
		
