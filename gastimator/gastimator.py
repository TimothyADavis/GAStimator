#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

class gastimator:
  def __init__(self, model,*args, **kwargs):
      self.targetrate= 0.25 
      self.dec=0.95
      self.inc=1.05
      self.model=model
      self.silent=False
      self.rng=None
      self.args=args
      self.kwargs=kwargs
      self.min=None
      self.max=None
      self.guesses=None
      self.precision=None
      self.prior_func=None
      self.fixed=None
      self.fdata=None
      self.error=None
      self.labels=None
      self.change=None
      self.npars=None
      self.lastchain=None
      self.lastchainll=None
      self.lastchainaccept=None

  def likelihood(self,values):
    
    priorval=1
    for prior,val in zip(self.prior_func,values):
        if callable(prior):
            priorval*=prior(val)
    
    
    chi2=((self.fdata - self.model(values,*self.args,**self.kwargs))**2 / self.error**2).sum()
    
    return -0.5*chi2 + np.log(priorval, where=(priorval!=0))
    


  def take_step(self,values,changethistime, knob):
      newvals=values.copy()
      acceptable=False
      cnt=0.0
     

      while not acceptable:
          newvals[changethistime]=values[changethistime]+(self.rng.randn(1)*knob[changethistime])
          if (newvals[changethistime] >= self.min[changethistime]) * (newvals[changethistime] <= self.max[changethistime]): acceptable=True
          if cnt > 5000: raise Exception('Cannot find a good step. Check input parameters')
      return newvals
          

      

  def evaluate_step(self, oldvalues, oldll, changethistime, knob, holdknob=False):
      
      accept=True
      tryvals=self.take_step(oldvalues,changethistime, knob)
      
      newll=self.likelihood(tryvals) 
      
      
      ratio=  newll-oldll
      randacc=np.log(self.rng.uniform(0,1,1))
      
      if randacc > ratio:
          accept=False
          newval = oldvalues
          newll = oldll
          if not holdknob: knob[changethistime]*=self.dec
      else:
          newval=tryvals
          if (not holdknob) and (knob[changethistime] < (self.max[changethistime]-self.min[changethistime])*5.): 
              knob[changethistime]*=self.inc
              
      return newval, newll, accept, tryvals, knob 
  
  

  def chain(self, values, niter, knob, plot=True, holdknob=False, fig=None, ax=None):
     

      if plot:
          fig, ax=self.make_plots(niter)
          plt.pause(0.00001)
          
      gibbs_change=self.rng.randint(0,self.change.size,niter)

      outputvals=np.empty((self.npars,niter))
      outputll=np.zeros(niter)
      accepted=np.zeros(niter)
           
      
     
      bestll=self.likelihood(values)   
      outputvals[:,0]=values
      outputll[0]=bestll
      n_accept=0.
      
      for i in range(1,niter):
          newval, newll, accept, tryvals, knob =self.evaluate_step(outputvals[:,i-1],outputll[i-1],self.change[gibbs_change[i]],knob,holdknob=holdknob)
          if holdknob and not self.silent: self.bar.next()
          if plot:
              self.update_plot(fig, ax,self.change[gibbs_change[i]],i,tryvals, accept)

              
          outputvals[:,i]=newval
          outputll[i]=newll
          accepted[i]=accept
          n_accept+=accept
     
      acceptrate=np.float(n_accept)/niter
      

      
      if (holdknob):
         if (self.burn):
             outputvals= outputvals[:,int(self.burn):-1]
             outputll= outputll[int(self.burn):-1]
             accepted=accepted[int(self.burn):-1]
         outputvals=outputvals[:,accepted.astype(bool)]
         outputll=outputll[accepted.astype(bool)]
      
      if plot: plt.close()
      return outputvals, outputll, accepted, acceptrate, knob

  def factors(self,n):  
     facs=np.array(list(x for tup in ([i, n//i] for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup))
     facs=np.sort(facs)
     return facs[int(facs.size/2 -  1):int(facs.size/2 +1)]


  def update_plot(self, fig, ax, index, i, new_data, accept):
     plot2update=(ax.flat[index])
     if accept:
         plot2update.plot(i,new_data[index],'bo')
     else: plot2update.plot(i,new_data[index],'ro')
     fig.canvas.draw()
     fig.canvas.start_event_loop(0.000001)
     
            
            
  def make_plots(self, niter):
        nplots=self.npars
        plotfac=self.factors(nplots)
        

        fig, ax = plt.subplots(plotfac[1], plotfac[0], sharex='col')
       
            
        for i in range(0,nplots):
            (ax.flat[i]).set_ylabel(self.labels[i])
            (ax.flat[i]).set_xlim([0,niter])
            (ax.flat[i]).set_ylim([self.min[i],self.max[i]])
            
        plt.show(block=False)
        return fig, ax
        


        
  def run_a_chain(self,startguess,niters,numatonce,knob,plot=True,final=False):
        
        count=0
        converged=False
        oldmean=self.guesses*0.0
        newmean=0.0
        startpoint=startguess.copy()
        
        if not final:
            while (count < niters) and (not converged):

                outputvals, outputll, accepted, acceptrate, knob = self.chain(startpoint, numatonce, knob, plot=plot, holdknob=False)
                
                if acceptrate*numatonce > 1:
                    w,=np.where(accepted) 
                    startpoint=outputvals[:,w[-1]]
                    newmean=np.mean(outputvals[:,w],axis=1) 
                else: newmean=oldmean
            
                if count == 0:
                    self.lastchain=outputvals
                    self.lastchainll=outputll
                    self.lastchainaccept=accepted
                    if (not self.silent): print("     Chain has not converged - Accept rate: "+str(acceptrate))
                else:
                    self.lastchain=np.append(self.lastchain,outputvals,axis=1)
                    self.lastchainll=np.append(self.lastchainll,outputll)
                    self.lastchainaccept=np.append(self.lastchainaccept,accepted)
                    test=(np.abs(newmean-oldmean) < self.precision)
                    if test.sum() == test.size and (acceptrate >= self.targetrate):
                         converged=1      
                         if (not self.silent): print("Chain converged: LL: "+str(np.max(outputll))+" - Accept rate:"+str(acceptrate))
                    else:
                         if  (not self.silent): print("     Chain has not converged - Accept rate: "+str(acceptrate))
                         if test.sum() == test.size:
                             if  (not self.silent): print("Target rate not reached")
                         else: 
                             if  (not self.silent): print("     Still varying: "+str(self.labels[~test]))
                     
                oldmean=newmean    
                count += numatonce
            if not converged: print('WARNING: Chain did not converge in '+str(niters)+' steps')
           
                             
        else:
            if not self.silent:
                with Bar('Final chain', max=niters-1, suffix='%(percent)d%%') as self.bar:   
                     outputvals, outputll, accepted, acceptrate, knob = self.chain(startpoint, niters, knob, plot=False, holdknob=True)
            else:
                outputvals, outputll, accepted, acceptrate, knob = self.chain(startpoint, niters, knob, plot=False, holdknob=True)
           
        best_knob=knob
        return outputvals, outputll, best_knob
                    
  def input_checks(self):
    if np.any(self.guesses == None):
         raise Exception('Please set initial guesses')
    
    self.npars=self.guesses.size
         
    if np.any(self.fixed == None):
             self.fixed=np.zeros(self.npars, dtype=bool)
             print("You did not specify if any variables are fixed - I will continue assuming that none are")
        
    names=["minimum","maximum","fixed","precision","labels"]    
    check=[self.min,self.max,self.precision,self.labels]
    for x, nam in zip(check,names):
        if np.any(x == None):
             raise Exception('Please set parameter '+str(nam))
        else:
             if x.size != self.npars:
                 raise Exception('Number of constraints in '+str(nam)+' does not match number of parameters')
     
    if np.any(self.prior_func == None):
             self.prior_func=np.zeros(self.npars, dtype=bool)    
 
    
           
  def run(self, fdata, error, niters, numatonce=None, burn=None, nchains=1, plot=True, output=None, seed=None): 
    # check all required inputs set
    self.input_checks()

    # set up variables needed
    self.fdata=fdata
    self.error=error
    self.rng=np.random.RandomState(seed)
    self.change, = np.where(self.fixed == False)
    verybestvalues=self.guesses
    verybestknob=None
    verybestll=-1e31


    if not numatonce:  numatonce=50*self.npars
    if not burn:  
        self.burn=0.2*niters
    else:
        self.burn=burn


    for chainno in range(0,nchains):
        if not self.silent: print('Doing chain '+np.str(chainno+1))
        knob=(0.5*(self.max-self.min))
        outputvals, outputll, best_knob = self.run_a_chain(self.guesses,niters,numatonce,knob,plot=plot)
        if (np.max(outputll) > verybestll) or (chainno == 0):
            if not self.silent: print("Best chain so far!")
            w,=np.where(outputll == np.max(outputll))
            verybestvalues=outputvals[:,w[0]].reshape(self.npars) 
            verybestknob=best_knob
            verybestll=np.max(outputll)
            
            
    if not self.silent: print("verybestparam",verybestvalues)
    if not self.silent: print("Starting final chain")
    

    outputvalue, outputll, best_knob = self.run_a_chain(verybestvalues,niters,numatonce,verybestknob,final=True)
    
    if outputll.size < 1: 
        print('WARNING: No accepted models. Perhaps you need to increase the number of steps?')
    else:
        if not self.silent: 
            w,=np.where(outputll == np.max(outputll))
            print("verybestparam final",(outputvalue[:,w[0]]).reshape(self.npars))
        perc = np.percentile(outputvalue, [15.86, 50, 84.14], axis=1)
        sig_bestfit = (perc[2][:] - perc[0][:])/2.
        if not self.silent: print(sig_bestfit)
    return outputvalue, outputll
