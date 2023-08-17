"""
Code for writing surface evolver geometry files

@author: Nilay
email: nkumar4@nd.edu
Multicellular Systems Engineering Lab (MSELab)
Department of Chemical and Biomolecular Engineering
Institution: University of Notre Dame
"""
class GeometryWriter:
	import os
	ScriptDir = os.path.abspath(os.path.dirname(__file__)) #  this script's directory 
	# Surface evolver model parameter definition (to be initialized by user)
	parameters= ['PARAMETER squamous_apical', 'PARAMETER squamous_basal','PARAMETER squamous_lateral', 
	'PARAMETER cuboidal_apical', 'PARAMETER cuboidal_basal', 'PARAMETER cuboidal_lateral', 
	'PARAMETER columnar_apical', 'PARAMETER columnar_basal', 'PARAMETER columnar_lateral', 'PARAMETER squamous_cuboidal', 'PARAMETER cuboidal_columnar',  
	'PARAMETER k_squamous_apical', 'PARAMETER k_squamous_basal', 'PARAMETER k_squamous_lateral', 
	'PARAMETER k_cuboidal_apical', 'PARAMETER k_cuboidal_basal', 'PARAMETER k_cuboidal_lateral', 
	'PARAMETER k_columnar_apical', 'PARAMETER k_columnar_basal', 'PARAMETER k_columnar_lateral', 'PARAMETER k_squamous_cuboidal', 'PARAMETER k_cuboidal_columnar', 
	'PARAMETER l_squamous_apical', 'PARAMETER l_squamous_basal', 'PARAMETER l_squamous_lateral', 
	'PARAMETER l_cuboidal_apical', 'PARAMETER l_cuboidal_basal', 'PARAMETER l_cuboidal_lateral', 
	'PARAMETER l_columnar_apical', 'PARAMETER l_columnar_basal', 'PARAMETER l_columnar_lateral', 
	'PARAMETER l_squamous_cuboidal', 
	'PARAMETER l_cuboidal_columnar','PARAMETER k_ecm_basal', 'PARAMETER k_lumen_apical']

	# Specifying output directory for saving file
	outputDirectory = ScriptDir

	# Looking for geometrical attributes of the surface evolver initialization file that remains unchanged
	inputDirectory = ScriptDir
	inputFileName = f"{inputDirectory}/SEattributes.txt"


	def __init__(self,inputArr,paramPressure, outputFileName):		
		# Initialize the class with input array, system pressure, and output file name
		self.inpts = inputArr
		self.systemPressure = paramPressure
		self.outputFileName = f"{outputFileName}.fe"
		
		# Create and open the output file for writing
		with open(self.outputFileName,"w+") as wf:
			wf.write("\n")
			
		# Write parameters and features to the output file
		self.writeParameters()
		self.writeFeatures()

	def writeParameters(self):
		# Write parameters to the output file
		with open(self.outputFileName,"a+") as af:
			af.write("STRING\n")
			af.write("space_dimension 2\n\n")
			af.write("PRESSURE " + str(self.systemPressure) + "\n\n")
			for i,keys in enumerate(self.parameters):
				af.write(f"{keys} = {self.inpts[i]}\n")
			af.write("\n")

	def writeFeatures(self):
		Write = False
		with open(self.inputFileName,"r") as rf:
			# Iterate through lines in the input file
			for line in rf:
				if line == "PARAMETER hooke2_power = 4\n" :
					Write = True
				if Write == True :
					with open(self.outputFileName,"a+") as af:
						af.write(line)

