''' vamodels.py
Teste
This module implement a function that calculates the saliency map proposed by Itti, Koch and Neibur in 
REFs.
'''
__version__ = "$Revision$"
# $Source$
__license__ = "Cecill-C"
__revision__ = " $Id: actor.py 1586 2009-01-30 15:56:25Z cokelaer $ "
__docformat__ = 'reStructuredText'
    

import numpy as np
import scipy.misc
import scipy.signal
import matplotlib.image as imge
import matplotlib.pyplot as plt 
import cv2


def orientation_pyramid( im,degrees = 0, L = 8, sigma = 1, lambd = 10, gamma =0.5 , psi = np.pi * 0.5 ):
    ''' 
    @brief Calculate the orientation (Gabor) pyramid

    @param im: input image
    @param degrees: orientation angle (in degrees)
    @param L: levels of the decomposition
    @param sigma: standard deviation of the Gaussian envelope (Gabor filter)
    @param lambd: sine wavelength (Gabor filter)
    @param gamma: Gabor filter aspect ratio
    @param psi: Gabor filter frequency shift
    
    @retval out: pyramid (list of images)
    '''
    imf = []
    imd = []
    out = []
    
    imTemp = np.copy( im )
    
    hGauss = np.array( [ [ 1.0, 4, 6, 4, 1 ], [ 4, 16, 24, 16, 4 ], [ 6, 24, 36, 24, 6 ], [ 4, 16, 24, 16, 4 ], [ 1, 4, 6, 4, 1 ] ] )
    hGauss /= hGauss.sum()
    
    theta = degrees * np.pi / 180
    gaborSize = 8 * sigma + 5 
    hGabor = cv2.getGaborKernel( ( gaborSize, gaborSize ), sigma, theta, lambd, gamma, psi)  # size ??

    for i in range( L + 1 ):
         imTemp2 = scipy.signal.convolve2d( imTemp, hGauss, mode = 'same', boundary = 'wrap' )
         imf.append( imTemp2  ) # colocar o nome correto
         tgtSize = ( (imTemp2.shape[0] + 1)/2, (imTemp2.shape[1] + 1)/2 )
         imTemp = cv2.resize(imTemp2,dsize = (tgtSize[1],tgtSize[0]),interpolation = 1)
         imd.append( imTemp  )
    
    for i in range( L ):
        imTemp  = imd[ i ] - imf[ i + 1 ]
        imTemp2 = scipy.signal.convolve2d( imTemp, hGabor, mode='same', boundary = 'wrap' )
        # --- test ---
        #print('=============================================================================')
        #print('Degrees: ', degrees, '; Level: ', i+1)
        #print('inTemp (difference image) min and max: ', imTemp.min(), imTemp.max())
        #print('inTemp2 (Gabor filtered imTemp) min and max: ', imTemp2.min(), imTemp2.max())
        #print('Dimensoes (imagem, filtro de Gabor): ', imTemp.shape, hGabor.shape)
        #print('=============================================================================')
        # ------
       
        out.append( imTemp2 )
    
    return out


def gaussian_pyramid( im, n ):  #(VERIFIED)
    '''
    @brief This function creates a Gaussian pyramid of n levels
    @detail The core is the OpenCV function *pyrDown*, which filters (Gaussian filter) and then downsample the input image by a factor of 2. 
    @param im: image from which the pyramid will be created
    @param n: number of pyramid levels
    
    @retval out: list that contains the subimages that form the pyramid 
    
    '''
    def image_gaussian_pyramid( iimg, n ):
        imp = []
        for i in range(n):
            tmp = cv2.pyrDown(iimg)
            iimg = tmp
            imp.append(tmp)
        return imp
        
    if (type(im) is type([])): # if it is a list
        out = []
        for j in range(len(im)):
            out.append(image_gaussian_pyramid(im[j],n))
    else:
        out = image_gaussian_pyramid( im, n )

    return out


def centre_surround_feature_map( pyrmd, center, delta ):  # VERIFIED
    ''' 
    @brief Calculate center-surround feature maps for Intensity and Orientation components
    
    @param pyrm: input pyramid 
    @param center: pyramid levels that are taken as centers (tuple, array, list )
    @param delta: pyramid level shifts to determine surroundings (tuple, array, list)
    
    @retval out: list of center-surround feature maps
    '''
    out = []
    for i in center:
        for j in delta:
            temp = cv2.resize(pyrmd[j+i-1],dsize = (pyrmd[i-1].shape[1],pyrmd[i-1].shape[0]),interpolation = 1)
            out.append(np.abs(pyrmd[i-1]-temp))
    return out

def centre_surround_colour_feature_map( Ap, Bp, center, delta ):  # VERIFIED
    ''' 
    @brief Calculate centre-surround feature maps for Colour components
    
    @param Ap, Bp: first and second colour pyramids
    @param center: pyramid levels that are taken as centers (tuple, array, list )
    @param delta: pyramid level shifts to determine surroundings (tuple, array, list)
    
    @retval out: list of center-surround feature maps
    '''
    out = []
    for i in center:
        for j in delta:
            temp1 = cv2.resize(Ap[j+i-1],dsize = (Ap[i-1].shape[1],Ap[i-1].shape[0]),interpolation = 1)
            temp2 = cv2.resize(Bp[j+i-1],dsize = (Bp[i-1].shape[1],Bp[i-1].shape[0]),interpolation = 1)
            out.append(np.abs( ( Ap[ i - 1 ] - Bp[ i - 1 ] ) - ( temp2 - temp1 ) ) )
    return out


def difference_of_gaussians_filtering_and_update(im,co=0.5,ci= 1.5,sigmao=2.0,sigmai=25.0,cte=0.02,loop=1,bound='wrap'):
    '''
    Implements a Difference of Gaussians (DoG) filtering, as described in *A saliency-based search mechanism for overt and covert shifts of visual attention* (Itti, Laurent and Koch, Christof; Vision Research, vol. 40, 10-12, pp. 1489-1506, 2000)

    Parameters:

    im: input image
    co: multiplicative constant of the first Gaussian of the DoG (default value: 0.5)
    ci: multiplicative constant of the second Gaussian of the DoG (default value: 1.5)
    sigmao: standard deviation of the first Gaussian of the DoG (default value: 2), in % of the input image width
    sigmai: standard deviation of the second Gaussian of the DoG (default value: 25), in % of the input image width
    cte: inhibitory constant term (default value: 0.02)
    loop: number of iteractions (default value: 1)
    bound: how to handle boundaries (options: 'wrap' (default), 'fill', and 'symm')
    
    Output:
    
    filtim: filtered image

    '''
    # filter
    h = np.zeros(im.shape) # create a zero matrix with the same dimensions of the input image
    Lx,Ly = h.shape        # number of rows and columns
    
    xv = np.arange( Lx )     # vectors to create the mesh
    yv = np.arange( Ly )
    cx = ( Lx - 1 ) / 2           # centres (x and y)
    cy = ( Ly - 1 ) / 2

    xv = xv - cx	   # shift in the mesh vectors
    yv = yv - cy

    xm,ym = np.meshgrid( xv, yv )	# creating the meshgrid

    sigmao = ( sigmao / 100 ) * Ly    # calculating standard deviations, considering image width
    sigmai = ( sigmai / 100 ) * Ly
    
    k1 = ( co**2 ) / ( 2 * np.pi * ( sigmao**2 ) ) # multiplicative terms: first and second gaussians
    k2 = ( ci**2 ) / ( 2 * np.pi * ( sigmai**2 ) )
    
    h = k1 * np.exp(-(xm**2+ym**2)/(2*(sigmao**2))) - k2 * np.exp(-(xm**2+ym**2)/(2*(sigmai**2))) # filter

    filtim = im / im.max()   # variable that contains the filtered image (in this step: just the normalisation of the input image)
    
    for i in range( loop ):
        map1 = scipy.signal.convolve2d( filtim, h, boundary=bound, mode='same' ) # DoG filtering 
        filtim = filtim + map1 - cte					   # image update
        
        mask = filtim > 0                # mask for elimination of negative terms
        if mask.max() == 0:
            filtmin = 0
            break
        else:
            filtim = mask * filtim		 # masking
            filtim = filtim / (filtim.max())#+0.001)	 # image normalisation

    return filtim #, h, xm, ym
#=======================================================================




def conspicuity_map( C, loops ):
    ''' 
    # @brief Calculate the conspicuity map from a given centre surround difference (orientation components, but it can be used for any componente, since the feature maps are organised into just one list) 
    #
    # @param C: list containing the feature maps from which the conspicuity maps will be calculated
    # @param 
    #
    # @retval out conspicuity map
    '''
    out = np.zeros( C[len(C)-1].shape )
    sizet = C[len(C)-1].shape # target size
    for i in range(len(C)):
        if (C[i].shape != sizet):
            temp = cv2.resize(C[i],dsize = (sizet[1],sizet[0]),interpolation = 1)
        else:
            temp = C[i]


        out += difference_of_gaussians_filtering_and_update( temp, loop = loops )
    out = difference_of_gaussians_filtering_and_update( out, loop = loops )
    return out

def smikn( img, lps = 1, centre = (2,3,4), delta=(3,4), verbose = 'off' ):
    '''
    # @brief Saliency map proposed by Itti, Koch and Niebur 
    # @detail 
    #
    # @param im: image (numpy array)
    # @param 
    #
    # @retval salMap: saliency map (numpy array)
    #
    '''
    print('========================================================== ')
    pyramid_levels = centre[-1] + delta[-1]
    
    # Spliting in rgb and converting to float  (VERIFIED)
    rt = img[:,:,0].astype(float) # temporary r 
    gt = img[:,:,1].astype(float) # temporary g 
    bt = img[:,:,2].astype(float) # temporary b 
    if verbose ==  'on':
        print('Spliting in rgb and converting to float: done.')
    # ------
    
    # Intensity image (VERIFIED)
    I = ( rt + gt + bt )/3
    if verbose ==  'on':
        print('Intensity image: done.')
    
    # Normalisation of rgb channels (VERIFIED)
    Mask = I > (0.1 * I.max()) # Intensity mask: used to select  rgb regions to be normalised (1/10 of maximum of the entire intensity image)
    preConst = 0.01 * I.max() # constant to prevent infinity values due to division
    r = ( rt / ( I + preConst ) ) * Mask 
    g = ( gt / ( I + preConst ) ) * Mask
    b = ( bt / ( I + preConst ) ) * Mask
    if verbose ==  'on':
        print('Normalisation of rgb channels: done.')
  
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
    if verbose ==  'on':
        print('Calculating R, G, B and Y components: done.')
    
    # --------------------------------------------------------------------------


    # Creating the Gaussian pyramids (VERIFIED)
    Ip   = gaussian_pyramid(I,pyramid_levels)                     #Intensity
    Rp   = gaussian_pyramid(R,pyramid_levels)                     # Red
    Gp   = gaussian_pyramid(G,pyramid_levels)                     # Green
    Bp   = gaussian_pyramid(B,pyramid_levels)                     # Blue
    Yp   = gaussian_pyramid(Y,pyramid_levels)                     # Yellow
    O0   = orientation_pyramid(I, degrees = 0,   L=pyramid_levels)  # Calculation of orientation components
    O45  = orientation_pyramid(I, degrees = 45,  L=pyramid_levels)  # Calculation of orientation components
    O90  = orientation_pyramid(I, degrees = 90,  L=pyramid_levels)  # Calculation of orientation components
    O135 = orientation_pyramid(I, degrees = 135, L=pyramid_levels)  # Calculation of orientation components
    if verbose ==  'on':
        print('Creating the Gaussian pyramids: done.')
    # ------

    # Calculation of Center-surround differences (VERIFIED)
    # - intensity
    Csdi = centre_surround_feature_map( Ip, centre, delta )
    # - color 
    # -- RG components
    Csdrg = centre_surround_colour_feature_map( Rp, Gp, centre, delta ) 
    # -- BY components
    Csdby = centre_surround_colour_feature_map( Bp, Yp, centre, delta )
    # -- Orientation components
    Csdor0   = centre_surround_feature_map( O0  , centre, delta )
    Csdor45  = centre_surround_feature_map( O45 , centre, delta )
    Csdor90  = centre_surround_feature_map( O90 , centre, delta )
    Csdor135 = centre_surround_feature_map( O135, centre, delta )
    Csdor =  Csdor0 + Csdor45 + Csdor90 + Csdor135 # list concatenation
    if verbose ==  'on':
        print('Calculation of Center-surround differences: done.')
    # ------
    
    # Conspicuity maps
    rounds = lps
    # - instensity
    CmI = conspicuity_map( Csdi, loops = rounds )
    # - color
    CmC = conspicuity_map( Csdrg+Csdby, loops = rounds )
    # - orientation
    CmO = conspicuity_map( Csdor, loops = rounds)
    if verbose ==  'on':
        print('Conspicuity maps: done.')
    
    # Saliency Map
    salMap =  (CmI + CmC + CmO ) / 3.0
    if verbose ==  'on':
        print('Saliency Map: done.')
        
    return(salMap)
