# -project.py
"""
Created on Thu Feb 18 22:10:00 2021

@author: Nilay
"""

# importing the flow library
import os
from flow import FlowProject
# Importing the geometry_writer class containing geometryWriter function that is used to write
# Surface Evolver initialization file
from geometry_writer import geometryWriter

# Checking if an operation has been executed by checking f the initialization file exists
@FlowProject.label
def SE_file_exist_check(job):
	return os.path.isfile(job.fn("wingDisc.fe"))


#a) write_SE_initialization_file has been defined as a function under the Flowproject class
# 1) The function writes geometry files for initializing SE simulations
# 2) operation decorated identifies the function as an operation while running the job
#b) A post condition is defined to check if the job has been executed
@FlowProject.operation
@FlowProject.post(SE_file_exist_check)
def write_SE_initialization_file(job):
	geometryWriter(job.sp.parameter_model, job.sp.parameter_pressure, job.fn(job.sp.output_file_name))


# Defining a project operation for simulating the output geometry files
@FlowProject.operation
@flow.with_job
@flow.cmd # It ensures that the functions returns a hell command through this decorator
def simulate_SE_file(job):
	# -x ensures that evolver stops after any error messages. 
	# Genral synatx: evolver [-adehimqwxy] [-f file] [-pN] [datafile]
	return "evolver -x wingdisc.fe"

	
if __name__ == '__main__':
	FlowProject().main()	