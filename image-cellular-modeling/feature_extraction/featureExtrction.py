# -*- coding: utf-8 -*-
"""
Code for extracting morphological features from surface evolver simulations
@author: Nilay

INPUT (txt filr):
Vertex positions output as a result of surface evolver simulation

OUTPUT (d dimensional array):
Following features are extracted:
    A) Cell height of each individual cell (n_total,1)
    B) EFD approximation of the shape of cross section (4,n_harmonics))
    C) Local Curvature (n_total-1,1)
    D) Angular approxinmation of curvature (n_columnar/4,1)
    total features (d): n_total + 4*n_harmonics + n_total -1 _ n_columnar/4
    
The code also plots and saves the tissue shape

"""
# Import python libraies. spatial_efd has been imported for extracting EFD coefficients
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spatial_efd
import math
 
# Read the data containing vertex positions and edge attributes
df1 = pd.read_excel('geometryfinal.xlsx')
v_id = df1['id']
vpos_x = df1['x']
vpos_y = df1['y']

df2 = pd.read_excel('log_edges.xlsx')
e_id = df2['eid']
e_v1 = df2['v1']
e_v2 = df2['v2']

# Define geometrical attributes
n_squamous = 20
n_columnar = 100
n_cuboidal = 10;
n_total = n_squamous + n_columnar + n_cuboidal;
n_edges_total = 3*n_total;

# Plotting the disc with the vertex and edge data
plot1 = plt.figure(1)
for i in range(n_edges_total):
    x = [vpos_x[e_v1[i]-1], vpos_x[e_v2[i]-1]]
    y = [vpos_y[e_v1[i]-1], vpos_y[e_v2[i]-1]]
    plt.plot(x,y,'black')
    plt.axes().set_aspect('equal', 'datalim')
plt.savefig("gc2.svg")
plt.show


## Extracting cell height of the tissue
cell_lengths = np.zeros(n_edges_total)
id_cell_lengths = np.zeros(n_edges_total)
for i in range(n_edges_total):
    x1 = vpos_x[e_v1[i]-1]
    y1 = vpos_y[e_v1[i]-1]
    x2 = vpos_x[e_v2[i]-1]
    y2 = vpos_y[e_v2[i]-1]
    cell_lengths[i] = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
    id_cell_lengths[i] = i
plot2 = plt.figure(2)
plt.plot(id_cell_lengths,cell_lengths)
plt.show


# EFD fit for the whole tissue gemetry
contour_basal_x = vpos_x[n_total:2*n_total-1]
contour_basal_y = vpos_y[n_total:2*n_total-1]
harmonic = 20
coeffs = spatial_efd.CalculateEFD(contour_basal_x, contour_basal_y, harmonic)
coeffs, rotation = spatial_efd.normalize_efd(coeffs, size_invariant=True)
xt, yt = spatial_efd.inverse_transform(coeffs, harmonic=harmonic)
plot3 = plt.figure(3)
plt.plot(xt,yt,'black')
plt.axes().set_aspect('equal', 'datalim')
plt.show


# Extracting curvature of basal surface
dx = np.gradient(contour_basal_x)
dy = np.gradient(contour_basal_y)
d2x = np.gradient(dx)
d2y = np.gradient(dy)
curvature = eval('abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
plot4 = plt.figure(4)
plt.plot(np.linspace(1, 129, num=129),curvature)
plt.show
plt.ylim((0,0.9))


# Extracting tpoplogical information about the columnar-basal surface of epitheila:
# cells at equual distances from the center are selected for increasing distance 
# from the center and the angle subtended by these points is measured around center.
angle_neighbors = np.zeros(int(n_columnar / 4))
id_angle_neighbors = np.zeros(int(n_columnar / 4))
basal_center_x = vpos_x[n_total + (n_squamous + n_cuboidal + n_columnar)/2]  
basal_center_y = vpos_y[n_total + (n_squamous + n_cuboidal + n_columnar)/2]  
for i in range(int(n_columnar / 4)):
    pt1_x = vpos_x[(n_total +(n_squamous + n_cuboidal + n_columnar)/2) + 1 + 2*i]
    pt1_y = vpos_y[(n_total +(n_squamous + n_cuboidal + n_columnar)/2) + 1 + 2*i]
    pt2_x = vpos_x[(n_total +(n_squamous + n_cuboidal + n_columnar)/2) - 1 - 2*i]
    pt2_y = vpos_y[(n_total +(n_squamous + n_cuboidal + n_columnar)/2) - 1 - 2*i]    
    v_1 = [pt1_x - basal_center_x, pt1_y - basal_center_y]
    v_2 = [pt2_x - basal_center_x, pt2_y - basal_center_y]
    uv_1 = v_1 / np.linalg.norm(v_1)
    uv_2 = v_2 / np.linalg.norm(v_2)
    dot_product = np.dot(uv_1, uv_2)
    angle_neighbors[i] = np.arccos(dot_product)
    id_angle_neighbors[i] = i
plot5 = plt.figure(5)
plt.plot(id_angle_neighbors,angle_neighbors,'black')
plt.show     






    
    
    

 



