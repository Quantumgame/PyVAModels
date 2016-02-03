'''
file test.py

brief Test file for the sm_ittikoch module detailes
author Ronaldo de Freitas Zampolo 
version 1.0
date 28.jan.2016  
'''


import numpy as np
import scipy.misc
import matplotlib.image as imge
import matplotlib.pyplot as plt 
import cv2

import vamodels as vam

# Load an color image - the input image will be provided
#  by the user when the function is called

path = 'images/'
image_name = 'teste2.jpg'

img = plt.imread(path+image_name)

imsm = vam.smikn( img, lps = 3, centre =(2,3,4), delta=(3,4) , verbose = 'on')


# --- Output ---
print('==================================== ')
print(' Image name:', image_name)
print(' Original image shape: ',img.shape)
print('==================================== ')

plt.figure()
plt.imshow(img,vmin = 0, vmax = 255)
plt.figure()
plt.imshow(imsm,cmap='gray')
plt.show()
