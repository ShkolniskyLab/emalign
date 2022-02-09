#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:33:25 2021

@author: yaelharpaz1
"""

import numpy as np
import math
from numpy import fft
from pyfftw.interfaces.numpy_fft import ifftn, ifft2
from numpy import linalg as LA
from dev_utils import npy_to_mat, mat_to_npy


def E3(deltax,rho,N,idx):  
    # Take as an input the estimated shift deltax and the measured phase
    # difference between the signals, and check how well the phases induced by
    # deltax agree with the measured phases rho. That is, return the difference
    # between the induced phases and the measured ones.
    #
    # Only the phases whose index is given by idx are considered.
    # The sum of the squared absolute value of E3, that is,
    #   sum(abs(E3(x,rhat(idx),(n-1)./2,idx)).^2)
    # is the L2 error in the phases. This is the expression we minimize (over
    # deltax) to find the relative shift between the images.
    #
    # Yoel Shkolnisky, January 2014.
    ll = np.fix(N/2)
    freqrng = np.arange(-ll,N-ll) 
    [X,Y,Z] = np.meshgrid(freqrng,freqrng,freqrng,indexing='ij')
    y = (np.exp(2*np.pi*1j*(X.T.ravel()[idx].T*deltax[0]+Y.T.ravel()[idx].T*deltax[1]+Z.T.ravel()[idx].T*deltax[2])/N)-rho)  
    return y

#%%
def dE3(deltax,rho,N,idx): 
    # dE3(j,deltax,rho,N,idx) compute the gradient of the function E3.
    # The result is an array of size numel(idx) by 3,where the first colums is
    # dE3/dx, the second is dE3/dy, and the third is dE3/dz.   
    ll = np.fix(N/2);
    freqrng = np.arange(-ll,N-ll)
    [X,Y,Z] = np.meshgrid(freqrng,freqrng,freqrng,indexing='ij')
    y = np.zeros((np.size(idx),3)).astype(complex)
    g = np.exp(2*np.pi*1j*(X.T.ravel()[idx].T*deltax[0]+Y.T.ravel()[idx].T*deltax[1]+Z.T.ravel()[idx].T*deltax[2])/N)
    y[:,0] = ((g*np.conjugate(E3(deltax,rho,N,idx))-np.conjugate(g)*E3(deltax,rho,N,idx))*2*np.pi*1j*X.T.ravel()[idx].T/N).reshape((np.size(idx)))
    y[:,1] = ((g*np.conjugate(E3(deltax,rho,N,idx))-np.conjugate(g)*E3(deltax,rho,N,idx))*2*np.pi*1j*Y.T.ravel()[idx].T/N).reshape((np.size(idx)))
    y[:,2] = ((g*np.conjugate(E3(deltax,rho,N,idx))-np.conjugate(g)*E3(deltax,rho,N,idx))*2*np.pi*1j*Z.T.ravel()[idx].T/N).reshape((np.size(idx)))
    return y

#%%
def d2E3(deltax,rho,N,idx): 
    # d2E3(j,deltax,rho,N,idx) compute the Hessian of the function E3. 
    ll = np.fix(N/2)
    freqrng = np.arange(-ll,N-ll)
    [X,Y,Z] = np.meshgrid(freqrng,freqrng,freqrng,indexing='ij')  
    y = np.zeros((np.size(idx),9)).astype(complex) #  The (symmetric matrix of the) Hessian is 
                           # is stored as a vector containing 
                           # d2x,  dxdy, dxdz 
                           # dydx, d2y,  dydz
                           # dzdx, dzdy, d2z
    a = 2*np.pi/N
    g = np.exp(1j*a*(deltax[0]*X.T.ravel()[idx].T+deltax[1]*Y.T.ravel()[idx].T+deltax[2]*Z.T.ravel()[idx].T))
    y[:,0] = ((-(a*X.T.ravel()[idx].T)**2)*(g*E3(deltax,rho,N,idx).conj()+g.conj()*E3(deltax,rho,N,idx))+2*(a*X.T.ravel()[idx].T)**2).reshape((np.size(idx)))*np.ones((np.size(idx)))
    y[:,1] = ((-(a**2)*X.T.ravel()[idx].T*Y.T.ravel()[idx].T)*(g*E3(deltax,rho,N,idx).conj()+g.conj()*E3(deltax,rho,N,idx))+2*(a**2)*X.T.ravel()[idx].T*Y.T.ravel()[idx].T).reshape((np.size(idx)))*np.ones((np.size(idx)))
    y[:,2] = ((-(a**2)*X.T.ravel()[idx].T*Z.T.ravel()[idx].T)*(g*E3(deltax,rho,N,idx).conj()+g.conj()*E3(deltax,rho,N,idx))+2*(a**2)*X.T.ravel()[idx].T*Z.T.ravel()[idx].T).reshape((np.size(idx)))*np.ones((np.size(idx)))
    y[:,3] = y[:,1]
    y[:,4] = ((-(a*Y.T.ravel()[idx].T)**2)*(g*E3(deltax,rho,N,idx).conj()+g.conj()*E3(deltax,rho,N,idx))+2*(a*Y.T.ravel()[idx].T)**2).reshape((np.size(idx)))*np.ones((np.size(idx)))
    y[:,5] = ((-(a**2)*Y.T.ravel()[idx].T*Z.T.ravel()[idx].T)*(g*E3(deltax,rho,N,idx).conj()+g.conj()*E3(deltax,rho,N,idx))+2*(a**2)*Y.T.ravel()[idx].T*Z.T.ravel()[idx].T).reshape((np.size(idx)))*np.ones((np.size(idx)))
    y[:,6] = y[:,2]
    y[:,7] = y[:,5]
    y[:,8] = ((-(a*Z.T.ravel()[idx].T)**2)*(g*E3(deltax,rho,N,idx).conj()+g.conj()*E3(deltax,rho,N,idx))+2*(a*Z.T.ravel()[idx].T)**2).reshape((np.size(idx)))*np.ones((np.size(idx)))
    return y    

#%%
def linmin(p,xi,rhat,n,idx,tol):
    # Given an n-dimensional point p and and n-dimensional direction xi, finds
    # a point pmin where the function f takes a minimum along the direction xi
    # from p. fmin is the value of f at pmin.
    # data (optional) is additional info provided to the function f. Use [] to
    # indicate no data.
    [ax, bx, cx] = mnbrak(p,p+xi,rhat,n,idx)
    [pmin,fmin] = brentvec(ax,bx,cx,rhat,n,idx,tol)
    return pmin, fmin

#%%
def mnbrak(ax,bx,rhat,N,idx):
    # Bracket a minimum of the function f using the initial abscissas ax and
    # bx.
    # The function searches in the downhill direction and returns three points
    # that bracket a minimum of the function. The function also returns the
    # values of f at these points. f is a handle to a scalar function that take
    # an n-dimenional point.
    #
    # ax and bx are the initial search points, each in dimension n. The
    # function performs a one dimensional search on the line between ax and bx.
    #
    # err=0 if everything went OK. err=1 if the bracketing points become to far
    # apart. This probably means that the function is deacresing.
    #
    # ax, bx, cx are the bracketing points. fa, fb, fc are the corresponding
    # function values.
    #
    # data (optional) is additional info provided to the function f. Use [] to
    # indicate no data.
    GOLD = 2/(3-math.sqrt(5))-1
    GLIMIT = 100
    TINY = 1.0e-20
    TLIMIT = 1e6
    n = len(ax)
    if len(bx) != n:
        raise ValueError("ax and bx must have same dimensions")
    #err = 0
    ta = 0     # Minimize on the line between ax and bx. ax is considered 0 (v0),
    tb = LA.norm(bx-ax)     # bx is considered 1, and the vector bx-ax is one step.   
    if tb == 0: # ax and bx are the same. No direction is given.
        cx = bx
        return ax, bx, cx    
    v0 = ax    # Once one-dimensional bracketing is determined, the corresponding 
    vec = bx-ax  # points in n-dimensional space are computed.
    vec = vec/LA.norm(vec)
    fa = func(ax,rhat,N,idx)
    fb = func(bx,rhat,N,idx)  
    if fb > fa: # Switch ta and tb so that we can go downhill in the direction from ta to tb.
        tmp = ta; ta = tb; tb = tmp;
        tmp = fa; fa = fb; fb = tmp;    
    tc = tb+GOLD*(tb-ta) # First guess for tc.
    fc = func(v0+tc*vec,rhat,N,idx)
    done = (fc >= fb)
    while (~done):    #Keep returning until we bracket.         
        if abs(tc) > TLIMIT:  # don't allow the farthest point to go too far
            #err = 1
            done = 1
            continue       
        r = (tb-ta)*(fb-fc)   # Compute u by parabolic extrapolation from 
        q = (tb-tc)*(fb-fa)   # ta, tb, and tc. Tiny is used to prevent division by zero.
        u = tb-((tb-tc)*q-(tb-ta)*r)/(2.0*sign(max(abs(q-r),TINY),q-r))
        ulim = tb+GLIMIT*(tc-tb)    
        if ((tb-u)*(u-tc) > 0.0): # u is between tb and tc: try it.
            fu = func(v0+u*vec,rhat,n,idx)
            if (fu < fc):  # Got a minimum between b and c
                ta = tb; tb = u;
                fa = fb; fb = fu;
                done = 1
                continue
            elif (fu > fb): # Got a minimum between a and u
                tc = u; fc = fu;
                done = 1
                continue
            u = tc+GOLD*(tc-tb) # Parabolic fit was of no use. Use default magnification.
            fu = func(v0+u*vec,rhat,N,idx)
        elif ((tc-u)*(u-ulim) > 0.0): # Parabolic fit is between c and its allowed limit
            fu = func(v0+u*vec,rhat,n,idx)
            if (fu < fc):
                tb = tc; tc = u; u = tc+GOLD*(tc-tb);
                fb = fc; fc = fu; fu = func(v0+u*vec,rhat,n,idx)
        elif ((u-ulim)*(ulim-tc) >= 0.0): # Limit parabolic u to maximum allowed value
            u = ulim
            fu = func(v0+u*vec,rhat,N,idx)
        else:
            u = tc+GOLD*(tc-tb) # Reject parabolic u, use default magnification.
            fu = func(v0+u*vec,rhat,N,idx)        
        ta = tb; tb = tc; tc = u; # Eliminate oldest point and continue.
        fa = fb; fb = fc; fc = fu;
        done = (fc >= fb)
    # convert from one-dimensional bracketing parameter along the line from ax
    # to bx, to bracketing points in n-dimensions.
    ax = v0+ta*vec
    bx = v0+tb*vec
    cx = v0+tc*vec   
    return ax, bx, cx
    
#%%        
def sign(a,b):
    if b >= 0:
        c = abs(a)
    else:
        c = -abs(a)
    return c

#%%
def brentvec(ax,bx,cx,rhat,N,idx,tol):
    # Like brent but ax, bx, and cx are colinear vectors, and f is a scalar
    # funtion on R^{n}.       
    ITMAX = 100            # Maximum allowed number of iterations.
    CGOLD = (3-math.sqrt(5))/2  # Golden ratio.
    ZEPS = 1.0e-10        # Protects against trying to achieve fractional 
                          # accuracy for a minimum that is exactly zero.
    EPS = 1.0e-14                         
    if isinstance(ax,np.single) or isinstance(bx,np.single) or isinstance(cx,np.single):   
        ZEPS = 1.0e-3; EPS=1.0e-6;    
    x0 = ax
    dvec = (bx-ax)
    if LA.norm(dvec) < EPS: # no direction is given. The minimum is the current point
        xmin = x0
        fmin = func(xmin,rhat,N,idx)
        return xmin, fmin
    dvec = dvec/LA.norm(dvec)
    if (abs(np.dot(cx-ax,dvec))/(LA.norm(dvec)*LA.norm(cx-ax))-1) > EPS:
        raise ValueError("ax, bx, cx must be colinear")   
    # convert ax, bx, and cx to time coordinates on the 1D coordinate system
    # with origin at x0 and positive direction dvec.
    ax = 0
    bx = np.dot(bx-x0,dvec)/LA.norm(dvec)**2
    cx = np.dot(cx-x0,dvec)/LA.norm(dvec)**2   
    a = min(ax,cx)
    b = max(ax,cx)       
    e = 0.0
    x = bx; w = bx; v = bx;    
    fx = func(point(x0,dvec,x),rhat,N,idx); fv = fx; fw = fx;   
    for iter in range(ITMAX):
        xm = 0.5*(a+b)
        tol1 = tol*abs(x)+ZEPS
        tol2 = 2.0*(tol1);
        if (abs(x-xm) <= (tol2-0.5*(b-a))):
            xmin = point(x0,dvec,x)
            fmin = fx           
            return xmin, fmin        
        if (abs(e) > tol1):
            r = (x-w)*(fx-fv)
            q = (x-v)*(fx-fw)
            p = (x-v)*q-(x-w)*r
            q = 2.0*(q-r)
            if (q > 0.0):
                p = -p
            q = abs(q)
            etemp = e
            #e = d
            if (abs(p) >= abs(0.5*q*etemp) or p <= q*(a-x) or p >= q*(b-x)):
                if x >= xm:
                    e = a-x
                else:
                    e = b-x
                d = CGOLD*(e)
            else:
                d = p/q
                u = x+d
                if (u-a < tol2 or b-u < tol2):
                    d = sign(tol1,xm-x)
        else:
            if x >= xm:
                e = a-x
            else:
                e = b-x
            d = CGOLD*(e)
        if abs(d) >= tol1:
            u = x+d
        else:
            u = x+sign(tol1,d)             
        fu = func(point(x0,dvec,u),rhat,N,idx)
        if (fu <= fx):
            if (u >= x): 
                a = x
            else:
                b = x 
            v = w; w = x; x = u;
            fv = fw; fw = fx; fx = fu;
        else:
            if (u < x): 
                a = u
            else:
                b = u 
            if (fu <= fw or w == x):
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            elif (fu <= fv or v == x or v == w):
                v = u
                fv = fu    
    raise Warning("Too many iterations in brent")
    xmin = point(x0,dvec,x)
    fmin = fx    
    return xmin, fmin

#%%
def point(x0,dvec,t):
    x = x0+t*dvec
    return x

#%%
def func(x, rhat, n, idx):
    return np.sum(np.abs(E3(x, rhat.T.ravel()[idx].T, n, idx)) ** 2)

#%%
def register_translations_3d(vol1,vol2):
    # REGISTER_TRANSLATIONS_3D  Estimate relative shift between two volumes.
    # register_translations_3d(vol1,vol2,refdx)
    #   Estimate the relative shift between two volumes vol1 and vol2 to subpixel
    #   accuracy. The function uses phase correlation to estimate the relative
    #   shift to within one pixel accuray, and then refines the estimation
    #   using Newton iterations.   
    #   Input parameters:
    #   vol1,vol2 Two volumes to register. Volumes must be odd-sized.
    #   refidx    Two-dimensional vector with the true shift between the images,
    #             used for debugging purposes. (Optional)
    #   Output parameters
    #   estidx  A two-dimensional vector of how much to shift vol2 to aligh it
    #           with vol1. Returns -1 on error.
    #   err     Difference between estidx and refidx.
    
    # Take Fourer transform of both volumes and compute the phase correlation
    # factors.
    hats1 = fft.fftshift(fft.fftn(fft.ifftshift(vol1))) # Compute centered Fourier transform
    hats2 = fft.fftshift(fft.fftn(fft.ifftshift(vol2)))
    rhat = hats1*np.conj(hats2)/(abs(hats1*np.conj(hats2)))
    rhat[np.isnan(rhat)] = 0
    rhat[np.isinf(rhat)] = 0   
    n = np.size(vol1,0)
    ll = np.fix(n/2)
    freqrng = np.arange(-ll,n-ll)   
    # Compute the relative shift between the images to to within 1 pixel
    # accuracy.
    # mm is a window function that can be applied to the volumes before
    # computing their relative shift. Experimenting with the code shows that
    # windowing does not improve accuracy.
    mm = 1 # Windowing does not improve accuracy.
    r = fft.fftshift(fft.ifftn(fft.ifftshift(rhat*mm))).real #**** seems like precision error in rhat affect the result
    ii = np.argmax(r)    
    # Find the center
    cX = np.fix(n/2) 
    cY = np.fix(n/2) 
    cZ = np.fix(n/2)     
    [sX,sY,sZ] = np.unravel_index(ii,np.shape(r))
    est_shift = [cX-sX,cY-sY,cZ-sZ]   
    # Refine shift estimation using Newton iterations
    MAXITER = 200 # Maximal number of Newton iterations
    eps = 1.0e-8  # Error to terminate Newton    
    if isinstance(vol1,float) or isinstance(vol2,float): 
        eps = 1.0e-5
    iter_num = 1      # Iteration number
    x = est_shift # Initialize Newton from the phase correlation estimated shifts.
    p = 1 # Newton step size   
    # Use only phases close the origin.
    radius = np.fix(n/2)*0.5
    [xx,yy,zz] = np.meshgrid(freqrng,freqrng,freqrng,indexing='ij')     
    idx = np.array(np.nonzero(xx.ravel()**2 + yy.ravel()**2 + zz.ravel()**2 < radius**2))
    # Note that we never use the top-most and left-most frequnecies of rhat
    # (the frequnecies which have no conjugate-symmetric conuterpart) since in
    # such a case, for n even, we would need to modify these values in the
    # estimated phases to be real. In other words, for n even, these
    # frequnecies do not contain the required information, so ignore them.
    
    # A function to evaluate the L2 error of the estimated shifts.
    lstol = 1.0e-5 # Line search tolerance.
    f = np.sum(abs(E3(x,rhat.T.ravel()[idx].T,n,idx))**2)
    ferr = abs(f)
    failed = 0
    while iter_num <= MAXITER and ferr > eps and LA.norm(p) > eps and failed == 0:
        df = np.sum(dE3(x,rhat.T.ravel()[idx].T,n,idx),axis=0)
        d2f = np.sum(d2E3(x,rhat.T.ravel()[idx].T,n,idx),0).reshape((3,3))     
        df = df.ravel()
        p = LA.solve(-d2f,df)           
        # Line search. Instead of taking a full Newton step from x in the
        # direction p, find the minimum of f starting from x in the direction
        # p. Note that using line search is not the most efficient way (see
        # Numerical Receipes).
        fold = f   
        xmin = x
        fmin = f
        try:
            [xmin,fmin] = linmin(x,p,rhat,n,idx,lstol)
        except:
            failed = 1
        x = xmin
        f = fmin   
        ferr = abs(f-fold)/np.max(abs(f),0)
        iter_num = iter_num + 1
    if failed:
        estdx = -1
    elif iter_num >= MAXITER:
        print('Did not converge')
        estdx = x
    else:
        # Two more iterations to polish the estimate       
        if ferr > 0:
            # There is a case which I don't understand in which I get an error
            # in  brentvec (line 36) "ax, bx, cx must be colinear". To avoid
            # that, I do run polishing iterations if ferr=0.           
            df = np.sum(dE3(x,rhat.T.ravel()[idx].T,n,idx),axis=0)
            d2f = np.sum(d2E3(x,rhat.T.ravel()[idx].T,n,idx),0).reshape((3,3))
            df = df.ravel()
            p = LA.solve(-d2f,df)
            [x,f] = linmin(x,p,rhat,n,idx,lstol)
            fold = f            
        estdx = x
        ferr = abs(f-fold)/np.max(abs(f),0)   
    return estdx
        