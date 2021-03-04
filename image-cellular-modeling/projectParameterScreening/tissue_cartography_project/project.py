# -project.py
"""
Created on Thu Feb 18 22:10:00 2021

@author: Nilay
"""
# importing the flow library
from flow import FlowProject
# Importing the geometry_writer class containing geometryWriter function that is used to write
# Surface Evolver initialization file
from geometry_writer import geometryWriter

# Checking if an operation has been executed by checking f the initialization file exists
def SE_file_exist_check(job):
	return os.path.isfile("wingDisc.fe")


#a) write_SE_initialization_file has been defined as a function under the Flowproject class
# 1) The function writes geometry files for initializing SE simulations
# 2) operation decorated identifies the function as an operation while running the job
#b) A post condition is defined to check if the job has been executed
@FlowProject.operation
@FlowProject.post(SE_file_exist_check)
def write_SE_initialization_file(job):
	geometryWriter(job.sp.parameter_model, job.sp.parameter_pressure, job.sp.output_file_name)
	
if __name__ == '__main__':
	FlowProject().main()	