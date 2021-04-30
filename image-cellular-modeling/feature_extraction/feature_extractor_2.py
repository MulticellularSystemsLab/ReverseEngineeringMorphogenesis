# -*- coding: utf-8 -*-
"""
Code for extracting morphological features from surface evolver simulations
@author: Nilay

"""
# Import python libraies. spatial_efd has been imported for extracting EFD coefficients
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spatial_efd
import math
 
class FeatureExtractor:
    """ Feature extraction from surface evolver output files/ simulated wing disk...
    
    Arguements:
        Vertex positions output as a result of surface evolver simulation
    
    Returns
    (d dimensional array):
    Following features are extracted:
        A) edge_length(): Cell height of each individual cell (n_total,1)
        B) tissue_efd_coeff(#harmonics): EFD approximation of the shape of cross section (4,n_harmonics))
        C) tissue_local_curvature(): Local Curvature (n_total-1,1)
        D) tissue_basal_angle_variation(): Angular approxinmation of curvature (n_columnar/4,1)
        total features (d): n_total + 4*n_harmonics + n_total -1 _ n_columnar/4
        Function plot_tissue_shape() also plots and saves the tissue shape
    """

    def __init__(self, geometry_data, edge_data, n_squamous = 20, n_columnar = 100, n_cuboidal = 10):
        """ Initialize with geometry and edge data
    
        Arguments:
            geometry_data: DataFrame or Excel filename containing geometry information
            edge_data: DataFrame or Excel filename containing edge information
            n_squamous:
            n_columnar:
            n_cuboidal:
        """

        # Save geometry data
        if type(geometry_data) is str:
            geometry_data = pd.read_excel(geometry_data)
               a1 = [];
               a2 = [];
   
               a3 = [];
               with open(geometry_data) as f:
                   next(f)
                   for line in f:
                       data = line.split()
                       a1.append(float(data[0]))
                       a2.append(float(data[1]))
                       a3.append(float(data[2]))
    
        self.v_id  = a1
        self.vpos_x = a2
        self.vpos_y = a3
    
        # Save edge data
        if type(edge_data) is str:
            edge_data = pd.read_excel(edge_data)
        
        self.e_id = edge_data['eid']
        self.e_v1 = edge_data['v1']
        self.e_v2 = edge_data['v2']
        
        # Define geometrical attributes
        self.n_squamous = n_squamous
        self.n_columnar = n_columnar
        self.n_cuboidal = n_cuboidal
        self.n_total = self.n_squamous + self.n_columnar + self.n_cuboidal
        self.n_edges_total = 3*self.n_total

    def plot_tissue_shape(self, filename="tissue_shape.png"):
        """ Plotting the disc with the vertex and edge data
        
        Arguments:
            filename: If string, save plot to file. If None, skip.
            
        Returns:
            Nothing
            
        Action:
            Creates plot
        
        """
        # Plotting the disc with the vertex and edge data
        plot1 = plt.figure(1)
        
        # Loop over all edges
        for i in range(self.n_edges_total):
        
            # Assemble coordinates
            x = [self.vpos_x[self.e_v1[i]-1], self.vpos_x[self.e_v2[i]-1]]
            y = [self.vpos_y[self.e_v1[i]-1], self.vpos_y[self.e_v2[i]-1]]
            
            # Plot edge
            plt.plot(x,y,'black')
        
        # Adjust and label axes
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel("x [nondimensional]")
        plt.ylabel("y [nondimensional]")
        
        if filename is not None:
            plt.savefig(filename)
        plt.show()
        

    def edge_length(self):
        """ Extracting edge lengths for each cell
        
        Arguements:
        
        Returns:
            an array containing edge length of all the cells in model
        
        Action:
            Extraction of the geometrical property, cell length, for each individual cell in the model.
        """
        # Initialiing the arrays for storing edge lengths
        cell_edge_lengths = np.zeros(self.n_edges_total)
        id_cell_lengths = np.zeros(self.n_edges_total)
        
        # Looping over all the cell edges
        for i in range(self.n_edges_total):
            
            # Getting x and y co-ordinates of each nodes in the edge
            x1 = self.vpos_x[self.e_v1[i]-1]
            y1 = self.vpos_y[self.e_v1[i]-1]
            x2 = self.vpos_x[self.e_v2[i]-1]
            y2 = self.vpos_y[self.e_v2[i]-1]
            
            # Calculating eucliean distance between the points
            cell_edge_lengths[i] = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
            id_cell_lengths[i] = i
        
        # plotting edge lengths
        plt.plot(id_cell_lengths,cell_edge_lengths)
        plt.xlabel("Edge id")
        plt.ylabel("Edge length [nondimensional]")
        plt.show()
        
        # Passing each edge length of model output as a return value of this function
        return cell_edge_lengths
        

    def tissue_efd_coeff(self, harmonic):
        """ EFD fit for the whole tissue geometry
        
        Arguements:
            number of harmonics or terms in the efd expanision
        
        Returns:
            an array containing EFD coefficients nor,alized against orientation and size
        
        Action:
            Extraction of the EFD based rotation and translation invariant shape descriptors.
        """
        
        # Extracting the nodes located in the basal surface of tissue (both squamous and columnar)
        contour_basal_x = self.vpos_x[self.n_total:2*self.n_total-1]
        contour_basal_y = self.vpos_y[self.n_total:2*self.n_total-1]
        
        #Defining total number of terms in fourier approximation of shape
        #harmonic = 20
        
        # Using spatial efd library to extract efd coefficients from tissue boundary points
        coeffs = spatial_efd.CalculateEFD(contour_basal_x, contour_basal_y, harmonic)
        
        # Normalizing the coefficients against rotation and size
        coeffs, rotation = spatial_efd.normalize_efd(coeffs, size_invariant=True)
        
        # Reverse EFD for plotting the normalized tissue shape
        xt, yt = spatial_efd.inverse_transform(coeffs, harmonic=harmonic)
        
        # Plotting normalized tissue shape
        plt.plot(xt,yt,'black')
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel("x [nondimensional]")
        plt.ylabel("y [nondimensional]")
        plt.show()
        
        # Passing coefficient as a return value of this function
        return coeffs
        
        
        
    def tissue_local_curvature(self):
        """ Extracts local tissue curvature along the exterior basal surface
        
        Arguements:
            
        
        Returns:
            an array containing local tissue curvature along the basal surface
        
        Action:
            The functions measures local tisue curvature along the basal surface
            Curvature is extimated using the formulae defined in the eval input arguement
        """
        
        # Extracting the nodes located in the basal surface of tissue (both squamous and columnar)
        contour_basal_x = self.vpos_x[self.n_total:2*self.n_total-1]
        contour_basal_y = self.vpos_y[self.n_total:2*self.n_total-1]
        
        # Calculating first derivative
        dx = np.gradient(contour_basal_x)
        dy = np.gradient(contour_basal_y)
        
        # Calculating second derivative
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        # Calculating curvature
        curvature = eval('abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
        
        # Plotting variation of curvature along the basal surface
        plt.plot(np.linspace(1, self.n_total-1, num=self.n_total-1),curvature)
        plt.xlabel("cell id ")
        plt.ylabel("Local curvature (1 / length)")
        plt.show()
        
        # Returning curvature values
        return curvature
        

    def tissue_basal_angle_variation(self):
        """ Extracting tpoplogical information about the columnar-basal surface of epitheila:
            cells at equual distances from the center are selected for increasing distance 
            from the center and the angle subtended by these points is measured around center
        
        Arguements:
            
        
        Returns:
            variation of angle formed by cells located at equal distances from the center is 
            returned as an output
        
        Action:
            Center of the basal epithelia is located. 
            Nodes of cells at varying distance from the center is calculated
            Angle between the three points is measure.
        """
        # Arrays are initialized
        angle_neighbors = np.zeros(int(self.n_columnar / 4))
        id_angle_neighbors = np.zeros(int(self.n_columnar / 4))
        
        # Center of the basement membrane is calculated
        basal_center_x = self.vpos_x[self.n_total + (self.n_squamous + self.n_cuboidal + self.n_columnar)/2]  
        basal_center_y = self.vpos_y[self.n_total + (self.n_squamous + self.n_cuboidal + self.n_columnar)/2]  
        
        # Nodes are selected at varying sistance from the ceneter
        for i in range(int(self.n_columnar / 4)):
            
            # x and y coordinates of the nodes are calculated
            pt1_x = self.vpos_x[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) + 1 + 2*i]
            pt1_y = self.vpos_y[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) + 1 + 2*i]
            pt2_x = self.vpos_x[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) - 1 - 2*i]
            pt2_y = self.vpos_y[(self.n_total +(self.n_squamous + self.n_cuboidal + self.n_columnar)/2) - 1 - 2*i]    
            
            # Line koining these nodes and the center are defined as vectors
            v_1 = [pt1_x - basal_center_x, pt1_y - basal_center_y]
            v_2 = [pt2_x - basal_center_x, pt2_y - basal_center_y]
            
            # Unit vectors are defined
            uv_1 = v_1 / np.linalg.norm(v_1)
            uv_2 = v_2 / np.linalg.norm(v_2)
            
            # Angle between the vectors is measured
            dot_product = np.dot(uv_1, uv_2)
            angle_neighbors[i] = np.arccos(dot_product)
            id_angle_neighbors[i] = i
        
        # plotting the variation in angle
        plt.plot(id_angle_neighbors,angle_neighbors,'black')
        plt.show()
        
        # returning the feature vector
        return angle_neighbors
"""
END OF FILE
"""