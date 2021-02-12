""" 
Code for initialization of surface evolver simulations
@Author: Nilay

INPUT: 1) A script containing the geometrical attributes as a text file containing
faces, edges, vertices and bodies along with the repoulksion subroutine and compound commapnds
2) Parameter values for initialization of the simulations

OUTPUT: Surface evolver simulation file

Author notes:
to change output directory : change the value inside {} in line 28
to change name of output file : change the value in line 29 between \\ and .txt
to change input directory: change the value inside {} in line 39
to change name of input file : change the value in line 40 between \\ and .txt
to use predefined array of inpts decomment line 26 and put the values and comment line 25 else it will take input from console

"""
import os
ScriptDir = os.path.abspath(os.path.dirname(__file__)) #  this script's directory 

#  Surface evolver Model parameter definitions
parameters= ['PARAMETER squamous_apical', 'PARAMETER squamous_basal','PARAMETER squamous_lateral', 
'PARAMETER cuboidal_apical', 'PARAMETER cuboidal_basal', 'PARAMETER cuboidal_lateral', 
'PARAMETER columnar_apical', 'PARAMETER columnar_basal', 'PARAMETER columnar_lateral', 
'PARAMETER k_squamous_apical', 'PARAMETER k_squamous_basal', 'PARAMETER k_squamous_lateral', 
'PARAMETER k_cuboidal_apical', 'PARAMETER k_cuboidal_basal', 'PARAMETER k_cuboidal_lateral', 
'PARAMETER k_columnar_apical', 'PARAMETER k_columnar_basal', 'PARAMETER k_columnar_lateral', 
'PARAMETER l_squamous_apical', 'PARAMETER l_squamous_basal', 'PARAMETER l_squamous_lateral', 
'PARAMETER l_cuboidal_apical', 'PARAMETER l_cuboidal_basal', 'PARAMETER l_cuboidal_lateral', 
'PARAMETER l_columnar_apical', 'PARAMETER l_columnar_basal', 'PARAMETER l_columnar_lateral', 
'PARAMETER squamous_cuboidal', 'PARAMETER k_squamous_cuboidal', 'PARAMETER l_squamous_cuboidal', 
'PARAMETER cuboidal_columnar', 'PARAMETER k_cuboidal_columnar', 'PARAMETER l_cuboidal_columnar','PARAMETER hooke2_power']

#inpts = [int(input()) for i in range(len(parameters))]  # inputs to wparameters from console input comment it to use predefinfed values
inpts =[0]*34   # decomment this to use predifened parameter values and put those value inside the sqwuare brackets

# Define the name of the output file
outputDirectory = ScriptDir
outputFileName = f"{outputDirectory}\\codeOutput.txt"  
with open(outputFileName,"w+") as wf:
	wf.write("\n\n")

# Step where parameters are declared in the surface evolver filr
def writeParameters():
	with open(outputFileName,"a+") as af:
		for i,keys in enumerate(parameters):
			af.write(f"{keys} = {inpts[i]}\n")
		af.write("\n")

# Reading the text file containing evolver file initialization attributes (geometry)
#for 130 cells
inputDirectory = ScriptDir
inputFileName = f"{inputDirectory}\\SEattributes.txt"

# Appending the geometry file to parameters defined
def writeFeatures():
	Write = False
	with open(inputFileName,"r") as rf:
		for line in rf:
			if line == "PARAMETER hooke2_power = 4\n" :
				Write = True
			if Write == True :
				with open(outputFileName,"a+") as af:
					af.write(line)
# Writing to output file
writeParameters()
writeFeatures()


