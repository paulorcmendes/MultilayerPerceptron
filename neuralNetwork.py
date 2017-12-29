# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:35:24 2017

@author: paulo
"""
import numpy as np

my_data = np.genfromtxt('iris.csv', delimiter=';')
#normalizing values
for i in range(4):
    my_data[:,i] = ( my_data[:,i] - min(my_data[:,i]) ) / (max(my_data[:,i]) - min(my_data[:,i]))

train = list(my_data[0:35])+ list(my_data[50:85])+ list(my_data[100:135])
test  = list(my_data[35:50])+ list(my_data[85:100])+ list(my_data[135:150])

print test.__len__()


