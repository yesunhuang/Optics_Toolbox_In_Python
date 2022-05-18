'''
Name: 
Desriptption: 
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: 
Author: YesunHuang
Date: 2022-05-18 19:18:16
'''
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
import abc
import cv2
import matplotlib.pyplot as plt
import time
from scipy import signal
from scipy.fft import fft2,ifft2
from scipy.fftpack import fftshift

#Some constants
EPSILON=1e-12


class WavePropagator(metaclass=abc.ABCMeta):
    '''the abstract base class for wave propagator'''

    @abc.abstractmethod
    def __init__(self,sizeN:int, interval:list, k:float, z:float=1.0):
        pass

    @abc.abstractmethod
    def cal_wavefront(self,U0:np.ndarray,z:float=None):
        pass

class RaySomSolver(WavePropagator):
    '''A solver for simulate diffraction via Rayleigh-Sommerfeld method'''

    def __init__(self,sizeN:int, interval:list, k:float, z:float=1.0):
        '''
        name: __init__ 
        fuction: initialize the class
        param {sizeN}: simulation size
        param {interval}: a list of sample interval [x_interval,y_interval]
        param {k}: wave number
        param {z}: the default z
        param {**kwargs}: other params
        '''            
        self.sizeN=sizeN
        assert sizeN % 2!=0, 'Please use odd number for size'
        self.interval=interval
        self.flag=False
        self.k=k
        self.z=z
        self.renorm=interval[0]*interval[1]
    
    def __generate_g(self,z:float):
        '''
        name: __defautGreenFunction
        fuction: Generate the defaut Green function
        param {z}: observation distance
        return {g_func}: the g function
        '''     
        self.flag=True   
        self.g_func=np.zeros((self.sizeN,self.sizeN),dtype=np.complex)
        halfN=(self.sizeN-1)//2
        #print(halfN)
        for i in range(-halfN,halfN+1):
            x=i*self.interval[0]
            for j in range(-halfN,halfN+1):
                y=j*self.interval[1]
                r=np.sqrt(x**2+y**2+z**2)
                self.g_func[i+halfN,j+halfN]=(-1.0/(2*pi))*\
                                (1j*self.k*np.sign(z)-1.0/r)*\
                                (np.exp(1j*self.k*r*np.sign(z))/r)*\
                                (z/r)
        return self.g_func

    def cal_wavefront(self,U0:np.ndarray,z:float=None):
        '''
        name:cal_wavefront 
        fuction: calculate the wavefront of xy plane at desired z 
        param {U0}: wavefront at z0=0
        param {z}: propagating distance
        return {Uz}: wavefront at z
        '''        
        assert U0.shape[0]==self.sizeN, 'Size mismatched!'
        if (not z==None) or (not self.flag):
            self.__generate_g(self.z)
        self.Uz=signal.fftconvolve(U0,self.g_func,mode='same')
        #self.Uz=fftshift(ifft2(fft2(U0)*fft2(self.g_func)))
        return self.Uz*self.renorm

class AnSpectSolver(WavePropagator):
    '''A solver for simulate diffraction via Angular Spectrum method'''

    def __init__(self,sizeN:int, interval:list, k:float, z:float=1.0):
        '''
        name: __init__ 
        fuction: initialize the class
        param {sizeN}: simulation size
        param {interval}: a list of sample interval [x_interval,y_interval]
        param {k}: wave number
        param {z}: the defaut z
        param {**kwargs}: other params
        '''            
        self.sizeN=sizeN
        assert sizeN % 2!=0, 'Please use odd number for size'
        self.interval=np.asarray(interval)
        self.k=k
        self.lam=2*pi/k
        self.z=z
        self.renorm=interval[0]*interval[1]
    

    def cal_wavefront(self,U0:np.ndarray,z:float=None):
        '''
        name:cal_wavefront 
        fuction: calculate the wavefront of xy plane at desired z 
        param {U0}: wavefront at z0=0
        param {z}: propagating distance
        return {Uz}: wavefront at z
        '''        
        assert U0.shape[0]==self.sizeN, 'Size mismatched!'
        if z==None:
            z=self.z
        A0=fftshift(fft2(U0))
        halfN=(self.sizeN-1)//2
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
        #TODO:The renormalization here might be incorrect.
        return self.Uz*self.renorm

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
        halfN=(self.sizeN-1)//2
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
        halfN=(self.sizeN-1)//2
        for i in range(-halfN,halfN+1):
            X=i*self.interval[0]
            for j in range(-halfN,halfN+1):
                Y=j*self.interval[1]
                if (abs(X)<=size[0]/2) and (abs(Y)<=size[1]/2):
                    self.patternMatrix[i+halfN,j+halfN]=1.0*self.modulator(X,Y)
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
        if self.pattern=='rect':
            assert len(size)==2, 'Not correct params\' number'
            return self.__generateRectangle(size)

class HelperFunctions:
    '''Some support functions'''

    @staticmethod
    def displace_2d(I,xylabels,interval,figureSize=(4,3),\
                    xylim:list=[],enableColorBar:float=False):
        fig, axes = plt.subplots(1,1,figsize=figureSize)
        X=np.linspace(interval[0][0],interval[0][1],I.shape[1])
        Y=np.linspace(interval[1][0],interval[1][1],I.shape[0])
        ax=axes.contourf(X,Y,I)
        if len(xylim)!=0:
            axes.set_xlim(xylim[0])
            axes.set_ylim(xylim[1])
        axes.set_xlabel(xylabels[0])
        axes.set_ylabel(xylabels[1])
        #show color bar
        if enableColorBar:
            fig.colorbar(ax)

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
        self.Um=None
    
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
        halfN=(sizeN-1)//2
        Uz=np.zeros((sizeN,sizeN),dtype=complex)
        for i in range(-halfN,halfN+1):
            X=i*interval[0]
            for j in range(-halfN,halfN+1):
                Y=j*interval[1]
                Uz[i+halfN,j+halfN]=U0[i+halfN,j+halfN]*self.modulator(X,Y)
        return Uz
    
    def get_modulator_matix(self,sizeN:int,interval:list):
        '''
        name: get_modulator_matix
        fuction: get the modulator matrix
        param {sizeN}: simulation size
        param {interval}: a list of sample interval [x_interval,y_interval]
        return {Um}: 2d array of modulator
        '''        
        if self.Um is not None:
            return self.Um

        halfN=(sizeN-1)//2
        self.Um=np.zeros((sizeN,sizeN),dtype=complex)
        for i in range(-halfN,halfN+1):
            X=i*interval[0]
            for j in range(-halfN,halfN+1):
                Y=j*interval[1]
                self.Um[i+halfN,j+halfN]=self.modulator(X,Y)
        return self.Um

class PhaseTypeHologram:
    '''A class built to calculate phase type hologram'''

    def __init__(self, sizeN:int, interval:list, pixelSize:list, shape:np.ndarray):
        '''
        name: __init__
        function: initialize the class
        param {sizeN}: sample size
        param {interval}: a list of sample interval [x_interval,y_interval]
        param {pixelSize}: the size of pixel
        param {shape}: determine the shape of hologram, the active pixel should not be zero. 
        It can also be the initial phase distribution of phase hologram. 
        '''
        self.sizeN=sizeN
        self.halfN=(sizeN-1)//2
        self.interval=interval
        self.pixelSize=pixelSize
        assert pixelSize[0]>=interval[0] and pixelSize[1]>=interval[1], 'pixelSize is too small'
        self.pixelIntervalN=[int(px/inl) for px,inl in zip(pixelSize,interval)]
        assert self.pixelIntervalN[0]<=self.sizeN and self.pixelIntervalN[1]<=self.sizeN, 'pixelInterval is too large'
        self.pixelN=[int(sizeN/pinI) for pinI in self.pixelIntervalN]
        self.shape=(np.real(shape)).astype(np.float32)
        assert shape.shape==(sizeN,sizeN), 'shape is not correct'
        self.hologramPhase=np.ones((self.pixelN[0],self.pixelN[1]),dtype=np.float32)*2*np.pi
        self.pixelShape=cv2.resize((self.shape).astype(np.float32),(self.pixelN[0],self.pixelN[1]),interpolation=cv2.INTER_NEAREST)
        self.hologramMatrix=np.exp(1j*cv2.resize(self.hologramPhase,(self.sizeN,self.sizeN),interpolation=cv2.INTER_NEAREST))

    def __call__(self,X,Y):
        '''
        name: __call__
        function: apply the modulator on (X,Y)
        param {X}: X coordinate
        paran {Y}: Y coordinate
        return {the modulator}
        '''
        return self.modulator(X,Y)

    def modulator(self,X,Y):
        '''
        name: modulator
        function: modulator the phase
        param {X}: X coordinate
        param {Y}: Y coordinate
        return {the phase value}
        '''
        x=int(X/self.interval[0])+self.halfN
        y=int(Y/self.interval[1])+self.halfN
        assert x>=0 and x<self.sizeN and y>=0 and y<self.sizeN, 'out of range'
        return self.get_hologram_matrix()[x,y]

    def __single_GS_epoch(self,incidentLight:np.ndarray,targetImage:np.ndarray,\
                        propagator:list):
        '''
        name: __single_GS_epoch
        param {incidentLight}: incident light distribution
        param {targetImage}: target image
        param {propagator}: a list of propagator
        '''
        #propagate forward
        U=incidentLight*self.get_hologram_matrix()
        U=propagator[0].cal_wavefront(U)
        ICurrent=HelperFunctions.intensity(U)
        ICurrent=ICurrent/np.max(np.abs(ICurrent))
        diff=ICurrent-targetImage
        loss=np.sum(diff*diff)/np.size(diff)
        newPhase=np.angle(U)
        #propagate backward
        U=np.exp(1j*newPhase)*np.sqrt(targetImage)
        U=propagator[1].cal_wavefront(U)
        newPhase=(np.angle(U)).astype(np.float32)
        self.hologramPhase=cv2.resize(newPhase,(self.pixelN[0],self.pixelN[1]),interpolation=cv2.INTER_NEAREST)
        self.hologramMatrix=np.exp(1j*cv2.resize(self.hologramPhase,(self.sizeN,self.sizeN),interpolation=cv2.INTER_NEAREST))
        return loss

    def get_hologram(self,z:float,k:float,
                    incidentLight:np.ndarray,\
                    targetImage:np.ndarray,epoches:int=10,\
                    method:str='RS',epochStep:int=1,printLoss:float=True):
        '''
        name: get_hologram
        function: get the desired hologram via GS algorithm
        param {z}: the distance to the desired image plane
        param {k}: the wave number
        param {incidentLight}: incident light distribution
        param {targetImage}: target image
        param {epoches}: the optimization epoches
        param {method}: the propagation method 'RS', 'ANG'
        return {the loss}
        '''
        assert incidentLight.shape==(self.sizeN,self.sizeN), 'incidentLight is not correct'
        assert targetImage.shape==(self.sizeN,self.sizeN), 'targetImage is not correct'
        assert z>0, 'z is not correct'
        targetImage=targetImage/np.max(np.abs(targetImage))
        loss=[]
        if method=='RS':
            propagator=[RaySomSolver(self.sizeN,self.interval,k,z),\
                RaySomSolver(self.sizeN,self.interval,k,-z)]
        elif method=='ANG':
            propagator=[AnSpectSolver(self.sizeN,self.interval,k,z),\
                AnSpectSolver(self.sizeN,self.interval,k,-z)]
        ts=time.time()
        for epoch in range(epoches):
            loss.append(self.__single_GS_epoch(incidentLight,targetImage,propagator))
            if printLoss:
                if (epoch+1)%epochStep==0:
                    te=time.time()
                    print(f'Epoch [{epoch+1}/{epoches}], loss: {loss[-1]:f}, time: {te-ts:f}s')
                    ts=time.time()
        return loss

    def apply_hologram(self,z:float,k:float,
                    incidentLight:np.ndarray,method:str='RS'):
        '''
        name: apply_hologram
        function: apply the desired hologram via GS algorithm
        param {z}: the distance to the desired image plane
        param {k}: the wave number
        param {incidentLight}: incident light distribution
        param {method}: the propagation method 'RS', 'ANG'
        return {target U}
        '''
        if method=='RS':
            propagator=RaySomSolver(self.sizeN,self.interval,k,z)
        elif method=='ANG':
            propagator=AnSpectSolver(self.sizeN,self.interval,k,z)
        U=incidentLight*self.get_hologram_matrix()
        return propagator.cal_wavefront(U)

    def get_hologram_phase_distribution(self):
        '''
        name: get_hologram_distribution
        function: get the hologram distribution
        return {the hologram distribution}
        '''
        return np.angle(self.hologramMatrix)*self.shape

    def get_hologram_matrix(self):
        '''
        name: get_hologram_matrix
        function: get the hologram matrix
        return {the hologram matrix}
        '''
        return self.hologramMatrix*self.shape

class ImageSys4f:
    '''A class for implement a 4f imaging system'''
        




