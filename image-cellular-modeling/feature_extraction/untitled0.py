# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 17:24:30 2021

@author: Nilay
"""

a1 = []
a2 = []
a3 = []

#with open('0_1_0_001.txt') as f:
#    for line in f:
#		  linex = line.strip()
#		        if line == '': 
#        data = line.split()
#        a1.append(float(data[0]))
#        a2.append(float(data[1]))
#        a3.append(float(data[2]))
        
		
		
with open('0_1_0_001.txt') as f:
	next(f)
	for line in f:
		data = line.split()
		a1.append(float(data[0]))
		a2.append(float(data[1]))
		a3.append(float(data[2]))