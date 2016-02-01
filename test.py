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

imsm = ikvam.sm( img, lps = 3)

# -------------------------
#for i in range(len(Csdor)):
plt.figure()
plt.imshow(img,vmin = 0, vmax = 255)
plt.figure()
plt.imshow(imsm,cmap='gray')
plt.show()
