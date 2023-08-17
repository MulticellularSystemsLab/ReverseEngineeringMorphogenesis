"""
Created on Thu Jul 21 09:46:55 2022

@author: Nilay Kumar
email: nkumar4@nd.edu
Multicellular Systems Engineering Lab (MSELab)
Department of Chemical and Biomolecular Engineering
Institution: University of Notre Dame
"""

def SampleParameters(paraminputsMaster,param_index_model,var1,var2, termid):
	
	"""
    Arguements:
        1. paraminputsMaster (list, int): List containing the original parameter values.
        2. param_index_model (list, int): Dictionary containing parameter indices for the model.
        3. var1 (int): First parameter variable.
        4. var2 (int): Second parameter variable.
        5. termid (int): Identifier for the parameter adjustment scenario.
                      0 - Both parameters increased by 70%.
                      1 - Both parameters decreased by 70%.
                      2 - First parameter increased, second parameter decreased by 70%.
                      3 - First parameter decreased, second parameter increased by 70%.
					  
    Opertions: 
	 Adjusts specified parameters for parameter sampling.

    This function facilitates the adjustment of selected parameters to perform parameter sampling
    for the purpose of computing the Hessian matrix.

    Returns:
        list: Adjusted parameter values after parameter sampling.
    """
	
	if termid == 0:
		# Create a copy of the original parameter values
		param_sim = paraminputsMaster
		param_sim[param_index_model[var1]] = param_sim[param_index_model[var1]] + 0.7*param_sim[param_index_model[var1]];
		param_sim[param_index_model[var2]] = param_sim[param_index_model[var2]] + 0.7*param_sim[param_index_model[var2]];
		
	elif termid == 1:
		# Create a copy of the original parameter values
		param_sim = paraminputsMaster
		param_sim[param_index_model[var1]] = param_sim[param_index_model[var1]] - 0.7*param_sim[param_index_model[var1]];
		param_sim[param_index_model[var2]] = param_sim[param_index_model[var2]] - 0.7*param_sim[param_index_model[var2]];
		
	elif termid == 2:
		# Create a copy of the original parameter values
		param_sim = paraminputsMaster
		param_sim[param_index_model[var1]] = param_sim[param_index_model[var1]] + 0.7*param_sim[param_index_model[var1]];
		param_sim[param_index_model[var2]] = param_sim[param_index_model[var2]] - 0.7*param_sim[param_index_model[var2]];
		
	elif termid == 3:
		# Create a copy of the original parameter values
		param_sim = paraminputsMaster
		param_sim[param_index_model[var1]] = param_sim[param_index_model[var1]] - 0.7*param_sim[param_index_model[var1]];
		param_sim[param_index_model[var2]] = param_sim[param_index_model[var2]] + 0.7*param_sim[param_index_model[var2]];
		
	return param_sim
		
		
		
		
		
