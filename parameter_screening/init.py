# init.py
"""
Created on Thu Feb 18 22:10:00 2021
@author: Nilay

paraminputs:
1. PARAMETER squamous_apical
2. PARAMETER squamous_basal: [0 0.001]
3. PARAMETER squamous_lateral
4. PARAMETER cuboidal_apical
5. PARAMETER cuboidal_basal: [0 0.001]
6. PARAMETER cuboidal_lateral
7. PARAMETER columnar_apical
8. PARAMETER columnar_basal: [0 0.01]
9. PARAMETER columnar_lateral
10. PARAMETER squamous_cuboidal
11. PARAMETER cuboidal_columnar
12. PARAMETER k_squamous_apical
13. PARAMETER k_squamous_basal
14. PARAMETER k_squamous_lateral
15. PARAMETER k_cuboidal_apical 
16. PARAMETER k_cuboidal_basal 
17. PARAMETER k_cuboidal_lateral 
18. PARAMETER k_columnar_apical: [0.000001 10]
19. PARAMETER k_columnar_basal: [0.0000001 10]
20. PARAMETER k_columnar_lateral: [0.0000001 10]
21. PARAMETER k_squamous_cuboidal 
22. PARAMETER k_cuboidal_columnar 
23. PARAMETER l_squamous_apical
24. PARAMETER l_squamous_basal
25. PARAMETER l_squamous_lateral 
26. PARAMETER l_cuboidal_apical 
27. PARAMETER l_cuboidal_basal 
28. PARAMETER l_cuboidal_lateral 
29. PARAMETER l_columnar_apical 
30. PARAMETER l_columnar_basal 
31. PARAMETER l_columnar_lateral 
32. PARAMETER l_squamous_cuboidal 
33. PARAMETER l_cuboidal_columnar 
34. PARAMETER k_ecm_basal: [0 0.01]
35. PARAMETER k_lumen_apical 

"""

# Importing libraries
import signac
import numpy as np
from smt.sampling_methods import LHS

# Range for sampled parameters
xlimits = np.array([[0, 0.001],[0, 0.001],[0, 0.01],[0.000001, 10],[0.000001, 10],[0.000001, 10],[0, 0.01]])
sampling = LHS(xlimits = xlimits)

# Defining numvber of samples
num_samples = 20

# Implementing latin hypercube sampling
x = sampling(num_samples)

# Creating configuration file for initialization of project
project = signac.init_project('tissue-cartography-project')

# predefining a input parameter list
paraminputs = [0,0.0001,0,0,0,0,0,0.001,0,0,0, 0.1,0.1,10,0.1,0.1,0.1,0.1,10,0.0001,0.001,0.001, 1,1,0.6,0.6,0.6,0.6,0.2,0.1,3,0.6,1.8, 0.001,0.001]
# Craeting statepoints as inputs tor the geometryWriter function.
for p in range(0,num_samples):
    # tension squamous basal
    paraminputs[1] = x[p,0]
    # tension cuboidal basal
    paraminputs[4] = x[p,1]
    # tension columnar basal
    paraminputs[7] = x[p,2]
    # k columnar apical
    paraminputs[17] = x[p,3]
    # k columnar basal
    paraminputs[18] = x[p,4]
    # k columnar lateral
    paraminputs[19] = x[p,5]
    # K_ECM
    paraminputs[33] = x[p,6]
    
    # defining jobs in signac
    sp = {'parameter_model': paraminputs, 'parameter_pressure': 0.001, 'output_file_name': 'wingDisc'}
    job = project.open_job(sp)
    job.init()


    	

	
