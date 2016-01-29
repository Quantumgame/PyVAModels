## @file test.py
#
# @brief Test file for the sm_ittikoch module
# @detailes
# @author Ronaldo de Freitas Zampolo 
# @version 1.0
# @date 28.jan.2016  
#
#

import numpy as np
import scipy.misc
import matplotlib.image as imge
import matplotlib.pyplot as plt 
import cv2

import sm_ittikoch as ikvam

# Load an color image - the input image will be provided
#  by the user when the function is called
img = plt.imread('teste2.jpg')

imsm = ikvam.sm(img)

# -------------------------
#for i in range(len(Csdor)):
plt.figure()
plt.imshow(img,vmin = 0, vmax = 255)
#plt.figure()
#    plt.imshow(Csdor[i],cmap='gray', vmin = 0, vmax = 255)
#plt.imshow(CmI,cmap='gray')
#plt.figure()
#plt.imshow(CmC,cmap='gray')
'''
plt.figure()
plt.imshow(imsm,cmap='gray',vmin = 0, vmax = 0.850*imsm.max())
plt.figure();
plt.imshow(imsm>(0.70*imsm.max()),cmap='gray',vmin = 0, vmax = 1)
'''
plt.show()
