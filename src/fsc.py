"""
Created on Sun Sep 25 21:49:00 2022

@author: yoel shkolnisky
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
    
def ndgrid (*xi):
    '''
    Implementation of MATLAB's ndgrid function.
    Similar to meshgrid by with a different output ordering
    
    Parameters
    ----------
    *xi : x1, x2,...,xn: array_like
        1-D arrays representing the coordinates of a grid.

    Returns
    -------
    
    X1, X2,...,xn ndarray
    ndgrid(x1,x2,...,xn) replicates the grid vectors x1,x2,...,xn to produce 
    an n-dimensional full grid.
    '''

    return np.meshgrid(*xi, indexing='ij')


def FSCorr(m1,m2): 
    '''
    Compute the fourier shell correlation between the 3D maps m1 and m2,
    which must be n x n x n in size with the same n, assumed to be even.
    
    The FSC is defined as
                sum{F1 .* conj(F2)}
    c(i) = -----------------------------
           sqrt(sum|F1|^2 * sum|F2|^2)
    where F1 and F2 are the Fourier components at the given spatial 
    frequency i. i ranges from 1 to n/2-1, times the unit frequency 1/(n*res) 
    where res is the pixel size.
    
    Examples:
    --------
    c=FSCorr(m1,m2)
    
    Parameters
    ----------
    m1,m2 : ndarray
        n x n x n array. n assumed to be even

    Returns
    -------
    c : 1-D vector with Fourier shell correlation coefficients computed as 
    described above. 
    
    '''
    
    # First, construct the radius values for defining the shells.
    n,n1,n2 = m1.shape
    ctr = (n + 1) / 2
    origin = np.transpose(np.array([ctr,ctr,ctr]))

    x,y,z = ndgrid(np.arange(1 - origin[0],n - origin[0]+1),np.arange(1 - origin[1],n - origin[1]+1),np.arange(1 - origin[2],n - origin[2]+1))
    R = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    eps = 0.0001

    # Fourier-transform the maps
    f1 = np.fft.fftshift(np.fft.fftn(m1))
    f2 = np.fft.fftshift(np.fft.fftn(m2))

    # Perform the sums
    d0 = R < 0.5 + eps
    c = np.zeros((int(np.floor(n / 2)),1))
    
    for i in np.arange(1,int(np.floor(n / 2)+1)).reshape(-1):
        d1 = R < 0.5 + i + eps
        ring = np.logical_xor(d1,d0)
        #r1 = np.multiply(ring,f1)
        #r2 = np.multiply(ring,f2)
        #num = np.real(sum(sum(sum(np.multiply(r1,np.conjugate(r2))))))
        #den = np.sqrt(sum(sum(sum(np.abs(r1) ** 2))) * sum(sum(sum(np.abs(r2) ** 2))))
        
        r1 = f1[ring]
        r2 = f2[ring]
        num = np.real(sum(np.multiply(r1,np.conjugate(r2))))
        den = np.sqrt(sum(np.abs(r1) ** 2) * sum(np.abs(r2) ** 2))
            
        c[i-1] = num / den
        d0 = d1
    
    return c
    
    
def fscres(fsc,cutoff):
    '''
    Find the resolution from an FSC curve.
    r=fscres(fsc,cutoff)    determines the index of the last bin of the
    given fsc curve that is above the the given cutoff. Intermediate values
    of the FSC curve are estimated using interpolation.

    Parameters
    ----------
    fsc : 1-D array
        Fourier shell correlation coefficients, computed using FSCorr
    cutoff : float
        Cutoff value to determine resolution. 

    Returns
    -------
    r : double
        Estimated resolution
        
    '''
    
    n = np.asarray(fsc).size
    r = n
    x = np.arange(1,n+1)
    xx = np.arange(1,n+0.01,0.01)
    fsc_interp = interp.PchipInterpolator(x,fsc)
    y = fsc_interp(xx)
    ii = next((i for i, val in enumerate(y) if val < cutoff),-1)

    if ii != -1 :
        r = xx[ii]

    return r
    
def plotFSC(vol1, vol2 ,cutoff = 0.143, pixelsize = 1.0): 
    '''
    Draw Fourier shell correlation curve and estimated resolution.

    plotFSC(vol1,vol2,cutoff,pixelsize)
    Draw the Fourier shell correlation curve for the volumes vol1 and vol2,
    and determine the resolution according the the given cutoff and
    pixelsize. The volumes vol1 and vol2 must be aligned.
    
    Example
    -------
    vol1 = mrcfile.open('map_ref.mrc')
    vol2 = mrcfile.open('map_aligned.mrc')
    cutoff=0.143
    resA, fig = plotFSC(vol1.data,vol2.data,cutoff,1.5)


    Algorithm
    ---------
    The resolution resA is determined as follows.
    # The derivation is given in 1-D, but it is the same for 3D, if the three
    dimesions of the volume are equal.
    Denote
    B   bandwidth of the signal
    fs  Maximal frequnecy
    T   Sampling rate (pixel size)
    N   Length of the signal
    j   The index of the last frequnecy bin where the correlation is above
    the cutoff value.
    
    1. The bandwidth B is given by B=2*fs, as the signal has frequencies
    in the range [-fs,fs].
    2. According to the Nyquist criterion, the samping frequency 1/T must
    satisfy 1/T>=B. The critical sampling rate is 1/T=B=2*fs.
    3. The width of each frequnecy bin is B/N=2*fs/N.
    4. The frequnecy of the bin at index j is fc=2*fs*j/N.
    5. The resolution is defined as resA=1/fc=N/(2*fs*j).
    Due to 2 above, resA=N*T/j.
    6. FSCorr returns correlation values only for positive freqnecuies,
    that is is returns the correlation at n frequnecies such that N=2*n
    (it assumes that N is even). Thus, in terms n, resA is given by
    resA = 2*n*T/j = (2*T)*(n/j).
    
    Parameters
    ----------
    vol1, vol2 : ndarray
        3-D arrays of the same dimensions. Volumes must be aligned.
    cutoff : float, optional
        Correlation cutoff threshold to determine resolution. The resolution 
        is determined by the first frequnecy where the correlation goes below 
        this value. Common values are 0.143 and 0.5. The default is 0.143.
    pixelsize : float, optional
        Pixel size of the volumes. The default is 1.0A.

    Returns
    -------
    resA : float
    Resolution of the structure in Angstrom according to the given cutoff 
    value.
    
    fig :  matplotlib.figure.Figure
    Handle to the FSC curve plot.

    '''
    
    fsc = FSCorr(vol1,vol2)
    n = np.asarray(fsc).size
    
    fig, ax = plt.subplots(constrained_layout=True)
    ax.grid(visible = True, axis = 'both')
    ax.plot(np.arange(1,n+1),fsc,'-g', linewidth = 2.0)
    
    plt.xlim(np.array([1,n]))
    plt.ylim(np.array([- 0.1,1.05]))
    
    # Plot cutoff line
    y = np.ones((n)) * cutoff
    ax.plot(np.arange(1,n+1),y, color='b', linestyle='--', linewidth=1.5)
    
    # Compute resolution - fscres return the bin number where the cutoff
    # resolution is obtained.    
    j = fscres(fsc,cutoff)
    resA = 2 * pixelsize * n / j
        
    yy = ax.get_ylim()
    ax.vlines(j,0,yy[1], colors='b', linestyles='dashed') 

    # Replace the default ticks with frequnecy values   
    xticks_locs = ax.get_xticks()    
    df = 1.0 / (2.0 * pixelsize * n)
    xticks = xticks_locs * df
    
    xticks_labels = []
    for e in xticks:
        xticks_labels.append('{0:7.3f}'.format(e))
    ax.set_xticks(xticks_locs,xticks_labels)        
    ax.set_xlabel('1/A')
    
    # Add top axis
    ax2 = ax.twiny()
    xticks_locs = ax.get_xticks()   
    xticks_labels = []
    for e in xticks:
        if e > 0:
            xticks_labels.append('{0:7.3f}'.format(1/e))
        else:
            xticks_labels.append(' ')
                        
    ax2.set_xticks(xticks_locs,xticks_labels)        
    ax2.set_xlabel('A')
       
    plt.title('Resolution={0:5.2f}A'.format(resA))
    plt.show()

    return resA, fig    


def plotFSC2(vol1a, vol2a , vol1b, vol2b, labels=None, 
             cutoff = 0.143, pixelsize = 1.0, figname=None): 
    '''
    Draw Fourier shell correlation curve and estimate resolution.
    
    Use this function to plot the Fourier shell correlations of two pairs on 
    volumes on the same axis. This can be used to compare the quality of two
    reconstructions. 
    
    Specifically, draw the Fourier shell correlation curve for the pairs of 
    volumes (vol1a,vol2a) and (vol1b, vol2b), and determine the resolution of 
    each pair according the the given cutoff and pixelsize. Each pair of 
    volumes must be aligned.
    
    Use the given cutoff value for determining resolution. Common values
    for cutoff are 0.143 and 0.5.  The resolution is determined by the 
    first frequnecy where the correlation goes below this value.

    Example
    -------
    vol1a=ReadMRC('./vol1a.mrc');
    vol2a=ReadMRC('./vol2a.mrc');
    vol1b=ReadMRC('./vol1b.mrc');
    vol2b=ReadMRC('./vol2b.mrc');
    cutoff=0.143;
    resA=plotFSC(vol1a.data,vol2a.data,vol1b.data,vol2b.data,cutoff,1.5); 


    Parameters
    ----------
    vol1a, vol2a, vol1b, vol2b : ndarray
        3-D arrays of the same dimensions. Volumes must be aligned.
    cutoff : float, optional
        Correlation cutoff threshold to determine resolution. The resolution 
        is determined by the first frequnecy where the correlation goes below 
        this value. Common values are 0.143 and 0.5. The default is 0.143.
    pixelsize : float, optional
        Pixel size of the volumes. The default is 1.0A.
    figname: string, optional
        If not None, the FSC plot would be saved to a file with this name.

    Returns
    -------
    resAa : float
        Estimated resolution of vol1a and vol2a
    resAb : float
        Estimated resolution of vol1b and vol2b
    fig :  matplotlib.figure.Figure
        Handle to the FSC curve plot.
        
        
    See plotFSC for more details.

    '''

    sz1a = vol1a.shape
    sz2a = vol2a.shape
    sz1b = vol1b.shape
    sz2b = vol2b.shape
    
    if (sz1a != sz2a) or (sz1a != sz1b) or (sz1a != sz2b):
        raise ValueError('Dimensions of all input volumes must be the same')

    if (labels is not None) and (len(labels)!=2) :
        raise ValueError('labels must contain two labels')

    fsc_a = FSCorr(vol1a,vol2a)
    n = np.asarray(fsc_a).size
    fsc_b = FSCorr(vol1b,vol2b)
    
    fig, ax = plt.subplots(constrained_layout=True)
    ax.grid(visible = True, axis = 'both')
    ax.plot(np.arange(1,n+1),fsc_a,'-g', linewidth = 2.0)
    ax.plot(np.arange(1,n+1),fsc_b,'-r', linewidth = 2.0)
    
    plt.xlim(np.array([1,n]))
    plt.ylim(np.array([- 0.1,1.05]))
    
    # Plot cutoff line
    y = np.ones((n)) * cutoff
    ax.plot(np.arange(1,n+1),y, color='b', linestyle='--', linewidth=1.5)
    
    # Compute resolution - fscres return the bin number where the cutoff
    # resolution is obtained.    
    j_a = fscres(fsc_a,cutoff)
    resAa = 2 * pixelsize * n / j_a
    j_b = fscres(fsc_b,cutoff)
    resAb = 2 * pixelsize * n / j_b
        
    yy = ax.get_ylim()
    ax.vlines(j_a,0,yy[1], colors='b', linestyles='dashed') 
    ax.vlines(j_b,0,yy[1], colors='b', linestyles='dashed') 

    # Replace the default ticks with frequnecy values   
    xticks_locs = ax.get_xticks()    
    df = 1.0 / (2.0 * pixelsize * n)
    xticks = xticks_locs * df
    
    xticks_labels = []
    for e in xticks:
        xticks_labels.append('{0:7.3f}'.format(e))
    ax.set_xticks(xticks_locs,xticks_labels)        
    ax.set_xlabel('1/A')
    
    # Add top axis
    ax2 = ax.twiny()
    xticks_locs = ax.get_xticks()   
    xticks_labels = []
    for e in xticks:
        if e > 0:
            xticks_labels.append('{0:7.3f}'.format(1/e))
        else:
            xticks_labels.append(' ')
                        
    ax2.set_xticks(xticks_locs,xticks_labels)        
    ax2.set_xlabel('A')
           
    if labels is None:      
        ax.legend(['{0:5.2f}A'.format(resAa), '{0:5.2f}A'.format(resAb)])
    else:
        ax.legend([labels[0]+' {0:5.2f}A'.format(resAa), 
                   labels[1]+' {0:5.2f}A'.format(resAb)])

    fig1 = plt.gcf()
    plt.show()
    plt.draw()

    if figname is not None:
        fig1.savefig(figname, dpi = 300)

    return resAa, resAb, fig    


def plotFSC3(vol1a, vol2a , vol1b, vol2b, vol1c, vol2c, labels=None,
             cutoff = 0.143, pixelsize = 1.0, figname=None): 
    '''
    Same as plotFSC2 but for three pairs of volumes
    '''

    sz1a = vol1a.shape
    sz2a = vol2a.shape
    sz1b = vol1b.shape
    sz2b = vol2b.shape
    sz1c = vol1c.shape
    sz2c = vol2c.shape
    
    if not all(x == sz1a for x in [sz2a,sz1b,sz2b,sz1c,sz2c]):
        raise ValueError('Dimensions of all input volumes must be the same')
        
    if (labels is not None) and (len(labels)!=3) :
        raise ValueError('labels must contain three labels')

    fsc_a = FSCorr(vol1a,vol2a)
    n = np.asarray(fsc_a).size
    fsc_b = FSCorr(vol1b,vol2b)
    fsc_c = FSCorr(vol1c,vol2c)
    
    fig, ax = plt.subplots(constrained_layout=True)
    ax.grid(visible = True, axis = 'both')
    ax.plot(np.arange(1,n+1),fsc_a,'-g', linewidth = 2.0)
    ax.plot(np.arange(1,n+1),fsc_b,'-r', linewidth = 2.0)
    ax.plot(np.arange(1,n+1),fsc_c,'-b', linewidth = 2.0)
    
    plt.xlim(np.array([1,n]))
    plt.ylim(np.array([- 0.1,1.05]))
    
    # Plot cutoff line
    y = np.ones((n)) * cutoff
    ax.plot(np.arange(1,n+1),y, color='b', linestyle='--', linewidth=1.5)
    
    # Compute resolution - fscres return the bin number where the cutoff
    # resolution is obtained.    
    j_a = fscres(fsc_a,cutoff)
    resAa = 2 * pixelsize * n / j_a
    j_b = fscres(fsc_b,cutoff)
    resAb = 2 * pixelsize * n / j_b
    j_c = fscres(fsc_c,cutoff)
    resAc = 2 * pixelsize * n / j_c
            
    yy = ax.get_ylim()
    ax.vlines(j_a,0,yy[1], colors='b', linestyles='dashed') 
    ax.vlines(j_b,0,yy[1], colors='b', linestyles='dashed') 
    ax.vlines(j_c,0,yy[1], colors='b', linestyles='dashed') 

    # Replace the default ticks with frequnecy values   
    xticks_locs = ax.get_xticks()    
    df = 1.0 / (2.0 * pixelsize * n)
    xticks = xticks_locs * df
    
    xticks_labels = []
    for e in xticks:
        xticks_labels.append('{0:7.3f}'.format(e))
    ax.set_xticks(xticks_locs,xticks_labels)        
    ax.set_xlabel('1/A')
    
    # Add top axis
    ax2 = ax.twiny()
    xticks_locs = ax.get_xticks()   
    xticks_labels = []
    for e in xticks:
        if e > 0:
            xticks_labels.append('{0:7.3f}'.format(1/e))
        else:
            xticks_labels.append(' ')
                        
    ax2.set_xticks(xticks_locs,xticks_labels)        
    ax2.set_xlabel('A')
     
    if labels is None:      
        ax.legend(['{0:5.2f}A'.format(resAa), '{0:5.2f}A'.format(resAb), 
                   '{0:5.2f}A'.format(resAc)])
    else:
        ax.legend([labels[0]+' {0:5.2f}A'.format(resAa), 
                   labels[1]+' {0:5.2f}A'.format(resAb), 
                   labels[2]+' {0:5.2f}A'.format(resAc)])
        
    fig1 = plt.gcf()
    plt.show()
    plt.draw()

    if figname is not None:
        fig1.savefig(figname, dpi = 300)

    return resAa, resAb, resAc, fig    
