## @package sm_ittikoch.py
#
# @brief Saliency map (Itti & Koch)
# @version 1.0
# @author Ronaldo de Freitas Zampolo
# @date 29.jan.2016


import numpy as np
import scipy.misc
import matplotlib.image as imge
import matplotlib.pyplot as plt 
import cv2

## 
# @brief  
#
# @param 
# @param 
#
# @retval 
# NEED TO BE REFACTORED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def OrientPyr( im,degree=0,N=8,sigma=5,lambd=10,gamma=0.5,psi=0 ):
    out = []
    size = 8*sigma +5
    for i in range(len(deg)):
        theta = deg[i]* np.pi / 180
        #getGaborKernel(ksize,sigma,theta,lambd,gamma,psi=CV_PI*0.5,ktype=CV_64F )
        h = cv2.getGaborKernel((size, size), sigma, theta, lambd, gamma, psi)
        
        h /= h.sum()
        # ---
        '''plt.figure()
        plt.imshow(h,cmap='Oranges')
        # ---'''
        temp = cv2.filter2D(im, -1, h)
        out.append(temp)
    return out

## 
# @brief This function creates a Gaussian pyramid of n levels
# @detail The core is the OpenCV function *pyrDown*, which filters (Gaussian filter) and then downsample the input image by a factor of 2. 
# @param im: image from which the pyramid will be created
# @param n: number of pyramid levels
#
# @retval out: list that contains the subimages that form the pyramid 
#
def GaussPyr( im, n ):  #(VERIFIED)

    def GaussPyrImg( iimg, n ):
        imp = []
        for i in range(n):
            tmp = cv2.pyrDown(iimg)
            iimg = tmp
            imp.append(tmp)
        return imp
        
    if (type(im) is type([])): # if it is a list
        out = []
        for j in range(len(im)):
            out.append(GaussPyrImg(im[j],n))
    else:
        out = GaussPyrImg( im, n )

    return out

## 
# @brief 
#
# @param 
# @param 
#
# @retval 
#
def NonLinNorm( im, k = 10 ):
    #k : number of divisions for each dimension
    M = 1.0 # maximum
    #nmax = 200 # number of other maximuns

    stpr = im.shape[0]/k # step (rows)
    stpc = im.shape[1]/k # step (columns)

    minim = im.min()
    maxim = im.max()

    temp = M / (maxim - minim) *( im - minim )
    mask = (temp!=M)

    temp = np.multiply(temp,mask)

    maxv = []

    for  i in range(0,im.shape[0],stpr):
        for  j in range(0,im.shape[1],stpc):

            if (i+stpr-1)>im.shape[0]:
                if (j+stpc-1)>im.shape[1]:
                    wdw = temp[i:im.shape[0],j:im.shape[1]]
                else:
                    wdw = temp[i:im.shape[0],j:j+stpc-1]
                    
            elif (j+stpc-1)>im.shape[1]:
                wdw = temp[i:i+stpr-1,j:im.shape[1]]
            else:
                wdw = temp[i:i+stpr-1,j:j+stpc-1]

            maxv.append(wdw.max())
    '''for i in range(nmax):
        indx = temp2.argmax()
        lin=indx/temp2.shape[1]
        col=indx - lin * temp2.shape[1]
        maxelm = temp2[lin,col]
        minim = temp2.min()
        maxim = temp2.max()
        maxv.append(maxelm)
        temp2[lin,col]= 0'''
    #print len(maxv)
    mean = np.average(maxv)
    return temp * ((M-mean)** 2)

## 
# @brief 
#
# @param 
# @param 
#
# @retval 
#
def CenterSurr( pyrmd, center, delta ):
    out = []
    for i in center:
        for j in delta:
            temp = cv2.resize(pyrmd[j+i-1],dsize = (pyrmd[i-1].shape[1],pyrmd[i-1].shape[0]),interpolation = 1)
            out.append(pyrmd[i-1]-temp)
    return out
## 
# @brief 
#
# @param 
# @param 
#
# @retval 
#
def CenterSurrC( Ap, Bp, center, delta ):
    out = []
    for i in center:
        for j in delta:
            temp1 = cv2.resize(Ap[j+i-1],dsize = (Ap[i-1].shape[1],Ap[i-1].shape[0]),interpolation = 1)
            temp2 = cv2.resize(Bp[j+i-1],dsize = (Bp[i-1].shape[1],Bp[i-1].shape[0]),interpolation = 1)
            out.append((Ap[i-1]-Bp[i-1])-(temp2-temp1))
    return out
## 
# @brief 
#
# @param 
# @param 
#
# @retval 
#
def CenterSurrO( Op, center, delta ):
    out = []
    for theta in range(len(Op)):
        for i in center:
            for j in delta:
                #print i,j,theta
                temp = cv2.resize(Op[theta][j+i-1],dsize = (Op[theta][i-1].shape[1],Op[theta][i-1].shape[0]),interpolation = 1)
                out.append(Op[theta][i-1]-temp)
    return out


## 
# @brief 
#
# @param 
# @param 
#
# @retval 
#
def ConspMapI( C ):
    out = np.zeros( C[len(C)-1].shape )
    sizet = C[len(C)-1].shape # target size
    for i in range(len(C)):
        if (C[i].shape != sizet):
            temp = cv2.resize(C[i],dsize = (sizet[1],sizet[0]),interpolation = 1)
        else:
            temp = C[i]
        out += NonLinNorm( temp )
    return out
## 
# @brief 
#
# @param 
# @param 
#
# @retval 
#
def ConspMapC( C1, C2 ):
    out = np.zeros( C1[len(C1)-1].shape )
    
    sizet = C1[len(C1)-1].shape # target size
    for i in range(len(C1)):
        if (C1[i].shape != sizet):
            temp1 = cv2.resize(C1[i],dsize = (sizet[1],sizet[0]),interpolation = 1)
            temp2 = cv2.resize(C2[i],dsize = (sizet[1],sizet[0]),interpolation = 1)
        else:
            temp1 = C1[i]
            temp2 = C2[i]
        out += NonLinNorm( temp1 ) + NonLinNorm( temp2 )
    return out
## 
# @brief 
#
# @param 
# @param 
#
# @retval 
#
def ConspMapO( C, center, delta ):
    outt = np.zeros( C[len(C)-1].shape )
    out = np.zeros( C[len(C)-1].shape )
    sizet = C[len(C)-1].shape # target size
    for i in range(len(C)):
        if (C[i].shape != sizet):
            temp = cv2.resize(C[i],dsize = (sizet[1],sizet[0]),interpolation = 1)
        else:
            temp = C[i]
        outt += NonLinNorm( temp )
        if (np.mod(i,len(center)*len(delta)) == (len(center)*len(delta) - 1)):
            out += NonLinNorm( outt )
            outt = np.zeros( C[len(C)-1].shape )
    return out

## 
# @brief Saliency map proposed by Itti, Koch and Niebur 
# @detail 
#
# @param im: image (numpy array)
# @param 
#
# @retval SM: saliency map (numpy array)
#
def sm( img ):
    # Spliting in rgb and converting to float  (VERIFIED)
    rt = img[:,:,0].astype(float) # temporary r 
    gt = img[:,:,1].astype(float) # temporary g 
    bt = img[:,:,2].astype(float) # temporary b 

    # Intensity image (VERIFIED)
    I = ( rt + gt + bt )/3

    # Normalisation of rgb channels (VERIFIED)
    Mask = I > (0.1 * I.max()) # Intensity mask: used to select  rgb regions to be normalised (1/10 of maximum of the entire intensity image)
    r = ( rt / I ) * Mask 
    g = ( gt / I ) * Mask
    b = ( bt / I ) * Mask
    
    # --- Tests ---
    #plt.figure()
    #plt.imshow( rt , cmap='gray', vmin=0, vmax = rt.max())
    #plt.figure()
    #plt.imshow( (rt/I) , cmap='gray', vmin=0, vmax = r.max())
    #plt.figure()
    #plt.imshow( r , cmap='gray', vmin=0, vmax = r.max())
    #plt.figure()
    #plt.imshow( I , cmap='gray',vmin=0, vmax = 255)
    #plt.figure()
    #plt.imshow( Imask , cmap='gray',vmin=0, vmax = 1)
    #plt.show()
    # ---
 
    # Calculating R, G, B and Y components (VERIFIED)
    R = r - ( g + b ) / 2 # R (red) component
    Mask = R > 0
    R = R * Mask

    G = g - ( r + b ) / 2 # G (green) component
    Mask = G > 0
    G = G * Mask

    B = b - ( g + r ) / 2 # B (blue) component
    Mask = B > 0
    B = B * Mask
    
    Y = ( r + g ) / 2 - np.absolute( r - g ) / 2 - b # Y (yellow) component
    Mask = Y > 0
    Y = Y * Mask
    
    # for test purpose only: comment or erase it after verification ------------
    #plt.figure()
    #plt.imshow(Y, vmin = 0, vmax = Y.max(), cmap = 'gray')
    #plt.figure()
    #plt.imshow(Mask, vmin = 0, vmax = 1, cmap = 'gray')
    #plt.figure()
    #plt.imshow(Y1, vmin = 0, vmax = Y.max(), cmap = 'gray')
    #plt.show()
    # --------------------------------------------------------------------------


    # Creating the Gaussian pyramids
    N = 8 # pyramid levels
    Ip   = GaussPyr(I,N)                  #Intensity
    Rp   = GaussPyr(R,N)                  # Red
    Gp   = GaussPyr(G,N)                  # Green
    Bp   = GaussPyr(B,N)                  # Blue
    Yp   = GaussPyr(Y,N)                  # Yellow
    O0   = OrientPyr(I, degrees = 0,  N)  # Calculation of orientation components
    O45  = OrientPyr(I, degrees = 45, N)  # Calculation of orientation components
    O90  = OrientPyr(I, degrees = 90, N)  # Calculation of orientation components
    O135 = OrientPyr(I, degrees = 135,N)  # Calculation of orientation components
   
    # for test purpose only: comment or erase it after verification ------------
    for i in range(N):
         plt.figure()
         plt.imshow(Op[i,0], vmin = 0, vmax =Op[i,0].max(), cmap = 'gray')
    #plt.figure()
    #plt.imshow(Mask, vmin = 0, vmax = 1, cmap = 'gray')
    #plt.figure()
    #plt.imshow(Y1, vmin = 0, vmax = Y.max(), cmap = 'gray')
    plt.show()
    # --------------------------------------------------------------------------

    # Calculation of Center-surround differences
    center = (2,3,4)
    delta = (3,4)
    # - intensity
    Csdi = CenterSurr(Ip, center, delta)
    # - color 
    # -- RG components
    Csdrg = CenterSurrC(Rp, Gp, center, delta)
    # -- BY components
    Csdby = CenterSurrC(Bp, Yp, center, delta)
    # -- Orientation components
    Csdor = CenterSurrO(Op, center, delta)

    # Conspicuity maps
    # - instensity
    CmI = ConspMapI( Csdi )
    # - color
    CmC = ConspMapC( Csdrg, Csdby )
    # - orientation
    CmO = ConspMapO( Csdor, center, delta )

    # Saliency Map
    SM = NonLinNorm( CmI ) + NonLinNorm( CmC ) + NonLinNorm( CmO )
    SM /= 3

    return(SM)
