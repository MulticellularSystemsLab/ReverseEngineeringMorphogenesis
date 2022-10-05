"""
Created on Thu Jul 21 09:46:55 2022

@author: Nilay
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
		
		
		
		
		
