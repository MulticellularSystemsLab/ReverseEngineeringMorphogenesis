# -project.py
"""
The code executes the job or surface evolver input files in this case generates by init.py

Created on Thu Feb 18 22:10:00 2021

@author: Nilay Kumar
email: nkumar4@nd.edu
Multicellular Systems Engineering Lab (MSELab)
Department of Chemical and Biomolecular Engineering
Institution: University of Notre Dame
"""

# importing the flow library
import os
import flow
from flow import FlowProject, directives
# Importing the geometry_writer class containing geometryWriter function that is used to write
# Surface Evolver initialization file
from dependencies.geometry_writer import GeometryWriter
from dependencies.feature_extractor_4 import FeatureExtractor

# Checking if an operation has been executed by checking f the initialization file exists
@FlowProject.label
def SE_file_exist_check(job):
	return os.path.isfile(job.fn("wingDisc.fe"))

@FlowProject.label
def output_file_exist_check(job):
	return os.path.isfile(job.fn("vertices.txt"))

#a) write_SE_initialization_file has been defined as a function under the Flowproject class
# 1) The function writes geometry files for initializing SE simulations
# 2) operation decorated identifies the function as an operation while running the job
#b) A post condition is defined to check if the job has been executed
@FlowProject.operation
@FlowProject.post(SE_file_exist_check)
def write_SE_initialization_file(job):
	GeometryWriter(job.sp.parameter_model, job.sp.parameter_pressure, job.fn(job.sp.output_file_name))


# Defining a project operation for simulating the output geometry files
@FlowProject.operation
@flow.with_job
@flow.cmd # It ensures that the functions returns a hell command through this decorator
def simulate_SE_file(job):
	# Fetching evolver installation from the src location. Running the file
	return "/home/nkumar4/Desktop/evolver_installation/src/evolver wingDisc.fe"

@FlowProject.operation
@FlowProject.post(output_file_exist_check)
def write_geometrical_features(job):
    fe = FeatureExtractor(job.fn("vertices.txt"),'input_data/log_edges.xlsx')
    efd_coeff, efd_coeff_norm, angle_rotated = fe.tissue_efd_coeff(20)
    job.document["length"] = fe.edge_length()
    job.document["e_f_d"] = efd_coeff
    job.document["e_f_d_norm"] = efd_coeff_norm
    job.document["e_f_d_rot"] = angle_rotated
    job.document["curvature"] = fe.tissue_local_curvature()
    
	
if __name__ == '__main__':
	FlowProject().main()	
