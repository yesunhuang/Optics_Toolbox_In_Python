'''
Name: DiffractionTool
Desriptption: This file implements all the tools for simulating diffraction
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: This is for the course "Fundamental of modern optics"
Author: YesunHuang
Date: 2022-03-22 16:25:04
'''

#import all the things we need
from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft2,ifft2
from scipy.fftpack import fftshift

class RaySomSolver:
    '''A solver for simulate diffraction via Rayleigh-Sommerfeld method'''

    def __init__(self,sizeN:int, interval:list, k:float, **kwargs):
        '''
        name: __init__ 
        fuction: initialize the class
        param {sizeN}: simulation size
        param {interval}: a list of sample interval [x_interval,y_interval]
        param {k}: wave number
        param {**kwargs}: other params
        '''            
        self.sizeN=sizeN
        assert sizeN % 2!=0, 'Please use odd number for size'
        self.interval=interval
        self.k=k
        self.kwargs=kwargs
    
    def __generate_g(self,z:float):
        '''
        name: __defautGreenFunction
        fuction: Generate the defaut Green function
        param {z}: observation distance
        return {g_func}: the g function
        '''        
        self.g_func=np.zeros((self.sizeN,self.sizeN),dtype=np.complex)
        halfN=round(self.sizeN/2)
        for i in range(-halfN,halfN+1):
            x=i*self.interval[0]
            for j in range(-halfN,halfN+1):
                y=j*self.interval[1]
                r=np.sqrt(x**2+y**2+z**2)
                self.g_func[i+halfN,j+halfN]=(-1.0/(2*pi))*(1j*self.k-1.0/r)*\
                                (np.exp(1j*self.k*r)/r)*\
                                (z/r)
        return self.g_func

    def cal_wavefront(self,U0:np.ndarray,z:float):
        '''
        name:cal_wavefront 
        fuction: calculate the wavefront of xy plane at desired z 
        param {U0}: wavefront at z0=0
        param {z}: propagating distance
        return {Uz}: wavefront at z
        '''        
        assert U0.shape[0]==self.sizeN, 'Size mismatched!'
        self.__generate_g(z)
        self.Uz=signal.fftconvolve(U0,self.g_func,mode='same')
        #self.Uz=fftshift(ifft2(fft2(U0)*fft2(self.g_func)))
        return self.Uz

class AnSpectSolver:
    '''A solver for simulate diffraction via Angular Spectrum method'''

    def __init__(self,sizeN:int, interval:list, k:float, **kwargs):
        '''
        name: __init__ 
        fuction: initialize the class
        param {sizeN}: simulation size
        param {interval}: a list of sample interval [x_interval,y_interval]
        param {k}: wave number
        param {**kwargs}: other params
        '''            
        self.sizeN=sizeN
        assert sizeN % 2!=0, 'Please use odd number for size'
        self.interval=np.asarray(interval)
        self.k=k
        self.lam=2*pi/k
        self.kwargs=kwargs
    

    def cal_wavefront(self,U0:np.ndarray,z:float):
        '''
        name:cal_wavefront 
        fuction: calculate the wavefront of xy plane at desired z 
        param {U0}: wavefront at z0=0
        param {z}: propagating distance
        return {Uz}: wavefront at z
        '''        
        assert U0.shape[0]==self.sizeN, 'Size mismatched!'
        A0=fftshift(fft2(U0))
        halfN=round(self.sizeN/2)
        anInterval=np.ones(2)*self.lam/(self.sizeN*self.interval)
        #print(anInterval)
        for i in range(-halfN,halfN+1):
            for j in range(-halfN,halfN+1):
                cosBeta=j*anInterval[1]
                cosAlpha=i*anInterval[0]
                sum=np.power(cosBeta,2)+np.power(cosAlpha,2)          
                cosGamma=np.sqrt((1-sum)+0j)
                A0[i+halfN,j+halfN]*=np.exp(1j*self.k*z*cosGamma)
        self.Uz=ifft2(A0)
        return self.Uz

class PatternGenerator:
    '''A class for generating regular patterns'''

    def __init__(self,sizeN:int,interval:list,**kwargs):
        '''
        name: __init__
        fuction: initialize the pattern generator
        param {sizeN}: simulation size
        param {interval}: a list of sample interval [x_interval,y_interval]
        param {kwargs}: other params
            param {pattern}: 
                'circle': for circle pattern
                'square': for square pattern
                'rect': for rectangle pattern
            param {modulator}:
                f(X,Y): modulate the phase

        '''        
        self.sizeN=sizeN
        assert sizeN % 2!=0,'Please use odd number for size'
        self.interval=interval
        if 'pattern' in kwargs:
            self.pattern=kwargs['pattern']
        else:
            self.pattern='circle'       
        if 'modulator' in kwargs:
            self.modulator=kwargs['modulator']
        else:
            self.modulator=lambda X,Y:1.0   
    
    def __generateCircle(self,size:list):
        '''
        name:  __generateCircle
        fuction: for generating circle pattern
        param {size}: a list of pattern parameters
        return {pattern}
        '''        
        self.patternMatrix=np.zeros((self.sizeN,self.sizeN),dtype=complex)
        halfN=round(self.sizeN/2)
        for i in range(-halfN,halfN+1):
            X=i*self.interval[0]
            for j in range(-halfN,halfN+1):
                Y=j*self.interval[1]
                if np.sqrt(X**2+Y**2)<=size:
                    self.patternMatrix[i+halfN,j+halfN]=1.0*self.modulator(X,Y)
        return self.patternMatrix.copy()

    def __generateRectangle(self,size:list):
        '''
        name: __generateRectangle
        fuction: for generating rectangle pattern
        param {size}: a list of pattern parameters
        return {pattern}
        '''    
        self.patternMatrix=np.zeros((self.sizeN,self.sizeN),dtype=complex)    
        halfN=round(self.sizeN/2)
        for i in range(-halfN,halfN+1):
            X=i*self.interval[0]
            for j in range(-halfN,halfN+1):
                Y=j*self.interval[1]
                if (abs(X)<=size[0]/2) and (abs(Y)<=size[1]/2):
                    self.pattern[i+halfN,j+halfN]=1.0*self.modulator(X,Y)
        return self.patternMatrix.copy() 
    
    def generate(self,size:list):
        '''
        name: generate 
        fuction: generate different pattern {0,1}
        param {size}: a list of pattern parameters
            'circle':[r]
            'square':[a]
            'rectangle':[a,b]
        return {pattern}: 2d narrray
        '''       
        if self.pattern=='circle':
            assert len(size)==1, 'Not correct params\' number'
            return self.__generateCircle(size)
        if self.pattern=='sqaure':
            assert len(size)==1, 'Not correct params\' number'
            size=size.append(size[0])
            return self.__generateRectangle(size)
        if self.pattern=='rectangle':
            assert len(size)==2, 'Not correct params\' number'
            return self.__generateRectangle(size)

class HelperFunctions:
    '''Some support functions'''

    @staticmethod
    def displace_2d(I,xylabels,interval,figureSize=(4,3),**kwargs):
        fig, axes = plt.subplots(1,1,figsize=figureSize)
        X=np.linspace(interval[0][0],interval[0][1],I.shape[1])
        Y=np.linspace(interval[1][0],interval[1][1],I.shape[0])
        axes.contourf(X,Y,I)
        if 'xylim' in kwargs:
            xylim=kwargs['xylim']
            axes.set_xlim(xylim[0])
            axes.set_ylim(xylim[1])
        axes.set_xlabel(xylabels[0])
        axes.set_ylabel(xylabels[1])

    @staticmethod
    def intensity(U):
        return  np.real(U*np.conj(U))

class PhaseModulator:
    '''A class for modulating the phase'''

    def __init__(self,modulator=lambda X,Y:1.0):
        '''
        name: __init__
        fuction: initialize the phase modulator
        param {kwargs}:
            f(X,Y): modulate the phase
        '''        
        self.modulator=modulator
    
    def __call__(self,X,Y):
        return self.modulator(X,Y)
    
    def get_normal_lens_modulator(self,k:float,f:float,r:float):
        '''
        name: get_normal_lens_modulator
        fuction: get the normal lens modulator
        param {k}: wave vector
        param {f}: focal length
        param {r}: radius
        return {modulator}: modulator
        '''        
        def modulator(X,Y):
            r2=X**2+Y**2
            rl2=r**2
            if r2<rl2:
                return np.exp(-1j*k/(2*f)*(X**2+Y**2))
            else:
                return 0
        self.modulator=modulator
        return modulator
    
    def apply_modulator(self,U0:np.ndarray,sizeN:int,interval:list):
        '''
        name: apply_modulator
        fuction: apply the modulator to the phase
        param {U0}: 2d array
        param {sizeN}: simulation size
        param {interval}: a list of sample interval [x_interval,y_interval]
        return {Uz}: 2d array
        '''        
        halfN=round(sizeN/2)
        Uz=np.zeros((sizeN,sizeN),dtype=complex)
        for i in range(-halfN,halfN+1):
            X=i*interval[0]
            for j in range(-halfN,halfN+1):
                Y=j*interval[1]
                Uz[i+halfN,j+halfN]=U0[i+halfN,j+halfN]*self.modulator(X,Y)
        return Uz



                




