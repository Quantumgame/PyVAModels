## @package sm_ittikoch.py
#
# @brief Saliency map (Itti & Koch)
# @version 1.0
# @author Ronaldo de Freitas Zampolo
# @date 29.jan.2016


import numpy as np
import scipy.misc
import scipy.signal
import matplotlib.image as imge
import matplotlib.pyplot as plt 
import cv2

## 
# @brief Calculate the orientation (Gabor) pyramid
#
# @param im: input image
# @param degrees: orientation angle (in degrees)
# @param L: levels of the decomposition
# @param sigma: standard deviation of the Gaussian envelope (Gabor filter)
# @param lambd: sine wavelength (Gabor filter)
# @param gamma: Gabor filter aspect ratio
# @param psi: Gabor filter frequency shift
#
# @retval out: pyramid (list of images)
# 
def OrientPyr( im,degrees = 0, L = 8, sigma = 1, lambd = 10, gamma =0.5 , psi = np.pi * 0.5 ):
    imf = []
    imd = []
    out = []
    
    imTemp = np.copy( im )
    
    hGauss = np.array( [ [ 1.0, 4, 6, 4, 1 ], [ 4, 16, 24, 16, 4 ], [ 6, 24, 36, 24, 6 ], [ 4, 16, 24, 16, 4 ], [ 1, 4, 6, 4, 1 ] ] )
    hGauss /= hGauss.sum()
    # --- test ---
    #HGauss = np.fft.fftshift(np.fft.fft2(hGauss,s=(255,255)))
    #plt.figure()
    #plt.imshow(np.abs(HGauss), cmap = 'gray' )
    # ------

    # --- teste --- 
    #degrees = 135
    # ------
    theta = degrees * np.pi / 180
    gaborSize = 8 * sigma + 5 
    hGabor = cv2.getGaborKernel( ( gaborSize, gaborSize ), sigma, theta, lambd, gamma, psi)  # size ??
    hGabor /= hGabor.sum()
    # --- test ---
    #HGabor = np.fft.fftshift(np.fft.fft2(hGabor,s=(255,255)))
    #plt.figure()
    #plt.imshow(np.abs(HGabor), cmap = 'gray' )
    #plt.show()
    # ------

    for i in range( L + 1 ):
         imTemp2 = scipy.signal.convolve2d( imTemp, hGauss, mode = 'same' )
         imf.append( imTemp2  ) # colocar o nome correto
         tgtSize = ( (imTemp2.shape[0] + 1)/2, (imTemp2.shape[1] + 1)/2 )
         # --- test ---
         #print(tgtSize, type(tgtSize))
         # ------
         #imTemp = scipy.misc.imresize( imTemp2, size = tgtSize )
         imTemp = cv2.resize(imTemp2,dsize = (tgtSize[1],tgtSize[0]),interpolation = 1)
         imd.append( imTemp  )
    
    for i in range( L ):
        out.append( scipy.signal.convolve2d( imd[ i ] - imf[ i + 1 ], hGabor, mode='same' ) )
        # --- test ---
        #plt.figure()
        #plt.imshow(out[i], cmap = 'gray' )
        # ------
    
    #plt.show()
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
# @brief Non linear normalisation 
#
# @param im: input map
# @param k: number of repeatitions
#
# @retval normalised map
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
# @brief Calculate center-surround feature maps for Intensity and Orientation components
#
# @param pyrm: input pyramid 
# @param center: pyramid levels that are taken as centers (tuple, array, list )
# @param delta: pyramid level shifts to determine surroundings (tuple, array, list)
#
# @retval out: list of center-surround feature maps
#
def CenterSurr( pyrmd, center, delta ):  # VERIFIED
    out = []
    for i in center:
        for j in delta:
            temp = cv2.resize(pyrmd[j+i-1],dsize = (pyrmd[i-1].shape[1],pyrmd[i-1].shape[0]),interpolation = 1)
            out.append(np.abs(pyrmd[i-1]-temp))
            # --- test ---
            plt.figure()
            plt.imshow(np.abs(pyrmd[i-1]-temp), cmap = 'gray' )
            plt.show()
            # ------
    return out

## 
# @brief Calculate centre-surround feature maps for Colour components
#
# @param Ap, Bp: first and second colour pyramids
# @param center: pyramid levels that are taken as centers (tuple, array, list )
# @param delta: pyramid level shifts to determine surroundings (tuple, array, list)
#
# @retval out: list of center-surround feature maps
#
def CenterSurrC( Ap, Bp, center, delta ):  # VERIFIED
    out = []
    for i in center:
        for j in delta:
            temp1 = cv2.resize(Ap[j+i-1],dsize = (Ap[i-1].shape[1],Ap[i-1].shape[0]),interpolation = 1)
            temp2 = cv2.resize(Bp[j+i-1],dsize = (Bp[i-1].shape[1],Bp[i-1].shape[0]),interpolation = 1)
            out.append(np.abs( ( Ap[ i - 1 ] - Bp[ i - 1 ] ) - ( temp2 - temp1 ) ) )
            # --- test ---
            #plt.figure()
            #plt.imshow(out[len(out)-1], cmap = 'gray' )
            #plt.show()
            # ------
    return out

## 
# @brief Calculate the conspicuity map from a given centre-surrounding difference
#
# @param C: centre-surrounding difference
#
# @retval out: conspicuity map
#
def ConspMapI( C ):
    out = np.zeros( C[len(C)-1].shape ) # matrix of zeros with the same size of scale 4 centre-surround difference
    sizet = C[len(C)-1].shape # target size
    for i in range(len(C)):
        if (C[i].shape != sizet):
            temp = cv2.resize(C[i],dsize = (sizet[1],sizet[0]),interpolation = 1)
        else:
            temp = C[i]
        out += NonLinNorm( temp ) #Verify this one !!! Pay attention to the  
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


    # Creating the Gaussian pyramids (VERIFIED)
    N = 8 # pyramid levels
    Ip   = GaussPyr(I,N)                     #Intensity
    Rp   = GaussPyr(R,N)                     # Red
    Gp   = GaussPyr(G,N)                     # Green
    Bp   = GaussPyr(B,N)                     # Blue
    Yp   = GaussPyr(Y,N)                     # Yellow
    O0   = OrientPyr(I, degrees = 0,   L=N)  # Calculation of orientation components
    O45  = OrientPyr(I, degrees = 45,  L=N)  # Calculation of orientation components
    O90  = OrientPyr(I, degrees = 90,  L=N)  # Calculation of orientation components
    O135 = OrientPyr(I, degrees = 135, L=N)  # Calculation of orientation components
    
    # Calculation of Center-surround differences (VERIFIED)
    center = (2,3,4)
    delta = (3,4)
    # - intensity
    Csdi = CenterSurr( Ip, center, delta )
    # - color 
    # -- RG components
    Csdrg = CenterSurrC( Rp, Gp, center, delta ) 
    # -- BY components
    Csdby = CenterSurrC( Bp, Yp, center, delta )
    # -- Orientation components
    Csdor0   = CenterSurr( O0  , center, delta )
    Csdor45  = CenterSurr( O45 , center, delta )
    Csdor90  = CenterSurr( O90 , center, delta )
    Csdor135 = CenterSurr( O135, center, delta )
    
    # Conspicuity maps
    # - instensity
    CmI = ConspMapI( Csdi )
    '''
    # - color
    CmC = ConspMapC( Csdrg, Csdby )
    # - orientation
    CmO = ConspMapO( Csdor, center, delta )

    # Saliency Map
    SM = NonLinNorm( CmI ) + NonLinNorm( CmC ) + NonLinNorm( CmO )
    SM /= 3
    '''
    SM = 0
    return(SM)
