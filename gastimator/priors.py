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
              
        def eval(self,x):
            x = (x - self.mu) / self.sigma
            return np.exp(-x*x/2.0) / np.sqrt(2.0*np.pi) / self.sigma