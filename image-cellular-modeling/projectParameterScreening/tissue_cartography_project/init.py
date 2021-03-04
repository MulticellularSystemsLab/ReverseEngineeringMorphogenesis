# init.py
"""
Created on Thu Feb 18 22:10:00 2021

@author: Nilay
"""

# Importing libraries
import signac

# Creating configuration file for initialization of project
project = signac.init_project('tissue-cartography-project')

# Craeting statepoints as inputs tor the geometryWriter function.
for p in range(1,10):
	# Vrying pressure of the model
	# Detaiks about the parameter_model can be found in geometry_writer.py
	sp = {'parameter_model': [0]*35, 'parameter_pressure': p, 'output_file_name': 'wingDisc'}
	job = project.open_job(sp)
	job.init()

	

	