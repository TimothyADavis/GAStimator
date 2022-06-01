#!/usr/bin/env python3
# coding: utf-8
import numpy as np

class priors:
    def __init__():
        pass
    
    class gaussian:  
        def __init__(self,mu,sigma):
            self.mu=mu
            self.sigma=sigma
              
        def eval(self,x,**kwargs):
            xs = (x - self.mu) / self.sigma
            return (-(xs*xs)/2.0) - np.log(2.5066282746310002*self.sigma) #returns ln(prior)
            
