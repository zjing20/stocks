# -*- coding: utf-8 -*-
"""
@author: zjing20
"""
import numpy as np
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt

solvers.options['show_progress'] = False

class MeanVariancePortfolio():
    
    def __init__(self, assetReturns, assetSigmas, corrMat):
        
        self.assetReturns = assetReturns
        self.assetSigmas = assetSigmas
        self.corrMat = corrMat
        
        self.covMat = np.multiply(np.multiply(self.assetSigmas,self.corrMat),self.assetSigmas.T)
        
        self.nRiskAssets = self.assetReturns.shape[0]
        
        self.lowerBound = None
        self.upperBound = None
        
        self.riskAversion = 3.5
        self.targetReturn = None
        self._mode = 'risk aversion'
        
        self.rateRiskFree = None

    @property
    def weights(self):
    
        """
            Given returns (as a column vector), sigmas (standard deviations, as a column vector), 
            correlation matrix, and target expected return of the portfolio/risk aversion, return
            optimal weights of (riskless asset [if exists] and) the risk assets.

            If constraints exist, input lower as a column vector of the lower bounds of the weights
            of risk assets, and upper as a column vector of the upper bounds of the weights of risk
            assets. 
            
            Here we impose no contrainsts on the riskless asset.
        """

        # solvers.qp(P,q,G,h,A,b)
        # minimize 1/2*x.T@P@x + q.T@x
        #      s.t G@x <= h
        #          A@x =  b
        
        if self.rateRiskFree is None:
            nAssets = self.nRiskAssets
        else:
            nAssets = self.nRiskAssets + 1
        
        # optimize given target return
        if self._mode == 'target return':
            
            if self.rateRiskFree is None:

                P = matrix(self.covMat)
    
                A = matrix(np.vstack((
                                        np.ones(nAssets).reshape(1,-1),
                                        self.assetReturns.T
                                        )))
            
            else:
                
                P = np.diag(np.zeros(nAssets))
                P[1:,1:] = self.covMat
                P = matrix(P)
                
                A = matrix(np.vstack((
                                        np.ones(nAssets).reshape(1,-1),
                                        np.hstack(([[self.rateRiskFree]],self.assetReturns.T))
                                        )))
                
            q = matrix(0.0,(nAssets,1))
            b = matrix([1.0,self.targetReturn])

        # optimize given risk aversion
        elif self._mode == 'risk aversion':
            
            if self.rateRiskFree is None:

                P = matrix(self.riskAversion*self.covMat)
                
                q = matrix(-self.assetReturns)
                
            else:
                
                P = np.diag(np.zeros(nAssets))
                P[1:,1:] = self.covMat
                P = matrix(self.riskAversion*P)
                
                q = matrix(np.vstack(([[-self.rateRiskFree]],-self.assetReturns)))
        
            A = matrix(np.ones(nAssets).reshape(1,-1))
            b = matrix([1.0])

        else:

            raise NameError("mode can only be `risk aversion` or `target return`")

        # if no constraints
        if self.lowerBound is None and self.upperBound is None:
            sol = solvers.qp(P,q,None,None,A,b)
            return np.array(sol['x'])

        # if constraints    
        if self.lowerBound is None:
            G = matrix(np.eye(nAssets))
            h = matrix(self.upperBound)

        elif self.upperBound is None:
            G = matrix(-np.eye(nAssets))
            h = matrix(self.lowerBound)

        else:
            G = matrix(np.vstack((
                                    np.eye(nAssets),
                                    -np.eye(nAssets)
                                    )))
            h = matrix(np.vstack((
                                    self.upperBound,
                                    -self.lowerBound
                                    )))

        sol = solvers.qp(P,q,G,h,A,b)
        return np.array(sol['x'])
    
    @property
    def r(self):
        if self.rateRiskFree is None:
            return (self.weights.T @ self.assetReturns).item()
        else:
            assetReturns = np.vstack(([[self.rateRiskFree]],self.assetReturns))
            return (self.weights.T @ assetReturns).item()
    
    @property
    def var(self):
        if self.rateRiskFree is None:
            return (self.weights.T @ self.covMat @ self.weights).item()
        else:
            covMat = np.diag(np.zeros(self.nRiskAssets+1))
            covMat[1:,1:] = self.covMat
            return (self.weights.T @ covMat @ self.weights).item()
    
    @property
    def vol(self):
        return self.var**0.5
    
    @property
    def weightsGlobalMinVar(self):
        if self.rateRiskFree is not None:
            return np.vstack((
                                [[1]],                
                                np.zeros(self.nRiskAssets).reshape(-1,1)
                                ))
        else:
            
            P = matrix(self.covMat)
            q = matrix(0.0,(self.nRiskAssets,1))
            
            A = matrix(np.ones(self.nRiskAssets).reshape(1,-1))
            b = matrix([1.0])
            
             # if no constraints
            if self.lowerBound is None and self.upperBound is None:
                sol = solvers.qp(P,q,None,None,A,b)
                return np.array(sol['x'])
    
            # if constraints    
            if self.lowerBound is None:
                G = matrix(np.eye(self.nRiskAssets))
                h = matrix(self.upperBound)
    
            elif self.upperBound is None:
                G = matrix(-np.eye(self.nRiskAssets))
                h = matrix(self.lowerBound)
    
            else:
                G = matrix(np.vstack((
                                        np.eye(self.nRiskAssets),
                                        -np.eye(self.nRiskAssets)
                                        )))
                h = matrix(np.vstack((
                                        self.upperBound,
                                        -self.lowerBound
                                        )))
    
            sol = solvers.qp(P,q,G,h,A,b)
            return np.array(sol['x'])
    
    @property
    def rGlobalMinVar(self):
        if self.rateRiskFree is not None:
            return self.rateRiskFree
        else:
            return (self.weightsGlobalMinVar.T @ self.assetReturns).item()
    
    @property
    def varGlobalMinVar(self):
        if self.rateRiskFree is not None:
            return 0
        else:
            return (self.weightsGlobalMinVar.T @ self.covMat @ self.weightsGlobalMinVar).item()
    
    @property
    def volGlobalMinVar(self):
        return self.varGlobalMinVar**0.5
    
    def updateLowerBound(self,lowerBound):
        """

        Parameters
        ----------
        lowerBound : Could set lowerBound = 'zero' for 0 lowerBounds for convenience.

        Returns
        -------
        None.

        """
        if self.rateRiskFree is None:
            if lowerBound == 'zero':
                self.lowerBound = np.zeros(self.nRiskAssets).reshape(-1,1)
            else:
                self.lowerBound = lowerBound
        else:
            if lowerBound == 'zero':
                self.lowerBound = np.zeros(self.nRiskAssets+1).reshape(-1,1)
                self.lowerBound[0,0] = -1000 # an arbitrarily small number
            elif lowerBound is not None:
                self.lowerBound = np.vstack(([[-1000]],lowerBound))                    
            else:
                self.lowerBound = None    
                
    def updateUpperBound(self,upperBound):
        """

        Parameters
        ----------
        upperBound : Could set upperBound = 'one' for 1 upperBounds for convenience.

        Returns
        -------
        None.

        """
        if self.rateRiskFree is None:
            if upperBound == 'one':
                self.upperBound = np.ones(self.nRiskAssets).reshape(-1,1)
            else:
                self.upperBound = upperBound
        else:
            if upperBound == 'one':
                self.upperBound = np.ones(self.nRiskAssets+1).reshape(-1,1)
                self.upperBound[0,0] = 1000 # an arbitrarily large number
            elif upperBound is not None:
                self.upperBound = np.vstack(([[1000]],upperBound))
            else:
                self.upperBound = None
    
    def updateRiskAversion(self,riskAversion):
        self.riskAversion = riskAversion
        self.targetReturn = None
        self._mode = 'risk aversion'
        
    def updateTargetReturn(self,targetReturn):
        self.targetReturn = targetReturn
        self.riskAversion = None
        self._mode = 'target return'
        
    def updateRiskFreeRate(self,rateRiskFree):
        
        self.rateRiskFree = rateRiskFree
        if rateRiskFree is None:
            if self.lowerBound is not None and self.lowerBound.shape[0] == self.nRiskAssets + 1:
                self.lowerBound = self.lowerBound[1:]
            if self.upperBound is not None and self.upperBound.shape[0] == self.nRiskAssets + 1:
                self.upperBound = self.upperBound[1:]
        else:
            if self.lowerBound is not None and self.lowerBound.shape[0] == self.nRiskAssets:
                self.lowerBound = np.vstack(([[-1000]],self.lowerBound))
            if self.upperBound is not None and self.upperBound.shape[0] == self.nRiskAssets:
                self.upperBound = np.vstack(([[1000]],self.upperBound))

    def _returnBounds(self):
        
        """
        Return min and max returns possible with no riskless assets and at least one constraint
        """
        
        # solvers.qp(P,q,G,h,A,b)
        # minimize 1/2*x.T@P@x + q.T@x
        #      s.t G@x <= h
        #          A@x =  b
        
        cMin = matrix(self.assetReturns)
        cMax = matrix(-self.assetReturns)
        
        A = matrix(np.ones(self.nRiskAssets).reshape(1,-1))
        b = matrix([1.0])
        
        if self.lowerBound is None:
            G = matrix(np.eye(self.nRiskAssets))
            h = matrix(self.upperBound)

        elif self.upperBound is None:
            G = matrix(-np.eye(self.nRiskAssets))
            h = matrix(self.lowerBound)

        else:
            G = matrix(np.vstack((
                                    np.eye(self.nRiskAssets),
                                    -np.eye(self.nRiskAssets)
                                    )))
            h = matrix(np.vstack((
                                    self.upperBound,
                                    -self.lowerBound
                                    )))
        
        solMin = solvers.lp(cMin,G,h,A,b)
        solMax = solvers.lp(cMax,G,h,A,b)
        
        rMin = np.array(solMin['x']).T @ self.assetReturns
        rMax = np.array(solMax['x']).T @ self.assetReturns
        
        return rMin, rMax
    
    def plot(self):
        
        VSHIFT = 0.065
        
        tempRiskAversion = self.riskAversion
        tempTargetReturn = self.targetReturn
        tempmode = self._mode
        
        tempRateRiskFree = self.rateRiskFree
        
        self.updateRiskFreeRate(None)
        
        vols = []
        rets = []
        
        if self.lowerBound is None and self.upperBound is None:
            
            for i in np.arange(0,max(self.assetReturns)*3,0.001):
                self.updateTargetReturn(i)
                vols.append(self.vol)
                rets.append(self.r)
                
        else:
            
            for i in np.arange(*self._returnBounds(),0.001):
                self.updateTargetReturn(i)
                vols.append(self.vol)
                rets.append(self.r)
            
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle('Portfolio Allocations', fontsize=12)
        plt.xlabel('Volatility', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        
        plt.plot(vols,rets,label='Minimum Variance Frontier') # plot frontier
        ylow, yhigh = plt.ylim() 
        plt.plot(self.volGlobalMinVar,self.rGlobalMinVar,'ro')  # plot GMV
        plt.annotate((f'  GMV Portfolio (w/o riskless)\n'
                      f'  ({self.volGlobalMinVar:.2f},{self.rGlobalMinVar:.2f})'),
                     (self.volGlobalMinVar,self.rGlobalMinVar-(yhigh-ylow)*VSHIFT))
        
        self.updateRiskFreeRate(tempRateRiskFree)
        
        if tempmode == 'target return':
            self.updateTargetReturn(tempTargetReturn)
        
        elif tempmode == 'risk aversion':
            self.updateRiskAversion(tempRiskAversion)
        
        if self.rateRiskFree is not None:
            plt.plot((0,self.vol,(yhigh-self.rateRiskFree)/(self.r-self.rateRiskFree)*self.vol),
                     (self.rateRiskFree,self.r,yhigh),
                     label='Capital Allocation Line') # plot CAL
        
        plt.plot(self.vol,self.r,'go') # plot optimal
        plt.annotate((f'  Optimal Portfolio\n'
                      f'  ({self.vol:.2f},{self.r:.2f})'),
                     (self.vol,self.r))
        
        plt.legend(loc='upper left')
        plt.show()
        
        return fig
    
    

    
    

        