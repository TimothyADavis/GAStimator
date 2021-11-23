#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from joblib import Parallel, delayed,cpu_count  
from joblib.externals.loky import get_reusable_executor

def lnlike(data,model,error):
  # default log-likelihood function
  chi2=np.nansum((data - model)**2 / error**2)      
  return -0.5*chi2
      
def unwrap_self(args, **kwarg):
    return gastimator.run_a_chain(*args, **kwarg)

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
      self.nprocesses=np.int(cpu_count())-1
      self.lnlike_func=lnlike

    
  def likelihood(self,values):
    
    priorval=1
    for ival,prior in enumerate(self.prior_func):
        if callable(prior):
            priorval*=prior(values[ival],allvalues=values,ival=ival)
    
    model=self.model(values,*self.args,**self.kwargs)
    return self.lnlike_func(self.fdata,model,self.error) + np.log(priorval, where=(priorval!=0))

    


  def take_step(self,values,changethistime, knob):
      newvals=values.copy()
      acceptable=False
      cnt=0.0
     

      while not acceptable:
          newvals[changethistime]=values[changethistime]+(self.rng.randn(1)*knob[changethistime])
          if (newvals[changethistime] >= self.min[changethistime]) * (newvals[changethistime] <= self.max[changethistime]): acceptable=True
          if cnt > 5000: raise Exception('Cannot find a good step. Check input parameters')
          cnt+=1
          
      return newvals
          

      

  def evaluate_step(self, oldvalues, oldll, changethistime, knob, holdknob=False):
      
      accept=True
      tryvals=self.take_step(oldvalues,changethistime, knob)
      
      newll=self.likelihood(tryvals) 
      if np.isnan(newll):
        print("Your function returned a Nan for the following parameters",tryvals)
        print("Attempting to continue...")
        ll=-1e20
      
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
  
  

  def chain(self, values, niter, knob, plot=True, holdknob=False, fig=None, ax=None,progress=False,progid=0):
     

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
      
      if progress and (progid==0):
          pbar = tqdm(total=(niter)*self.nprocesses)
          
      for i in range(1,niter):
          newval, newll, accept, tryvals, knob =self.evaluate_step(outputvals[:,i-1],outputll[i-1],self.change[gibbs_change[i]],knob,holdknob=holdknob)
          if progress and (progid==0): pbar.update(self.nprocesses)
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
      if progress and (progid==0):
          pbar.close()
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
        


        
  def run_a_chain(self,startguess,niters,numatonce,knob,plot=True,final=False,progid=0):
        count=0
        converged=False
        oldmean=self.guesses*0.0
        newmean=0.0
        startpoint=startguess.copy()
        
        if not final:
            while (count < niters) and (not converged):

                outputvals, outputll, accepted, acceptrate, knob = self.chain(startpoint, numatonce, knob, plot=plot, holdknob=False,progress=False)
                
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
                             if  (not self.silent): print("     Target rate not reached")
                         else: 
                             if  (not self.silent): print("     Still varying: "+str(self.labels[~test]))
                     
                oldmean=newmean    
                count += numatonce
            if not converged: print('WARNING: Chain did not converge in '+str(niters)+' steps')
           
                             
        else:
            #if (self.silent == False)&(self.nprocesses==1):
            #        outputvals, outputll, accepted, acceptrate, knob = self.chain(startpoint, niters, knob, plot=False, holdknob=True,progress=True,progid=progid)
            #else:
            self.rng=np.random.RandomState() #refresh the RNG
            outputvals, outputll, accepted, acceptrate, knob = self.chain(startpoint, niters, knob, plot=False, holdknob=True,progress=True,progid=progid)
            
        best_knob=knob
        return outputvals, outputll, best_knob
                    
  def input_checks(self):
    self.guesses=np.array(self.guesses)
    self.min=np.array(self.min)
    self.max=np.array(self.max)
    self.precision=np.array(self.precision)
    self.labels=np.array(self.labels)
    

    
                
    if np.any(self.guesses == None):
         raise Exception('Please set initial guesses')
    self.npars=self.guesses.size
    
    if np.all(np.array(self.prior_func) == None):
             self.prior_func=(np.zeros(self.npars, dtype=bool)).tolist()
    else:
        try:
            if len(self.prior_func) != self.npars:
                 raise Exception('Number of priors given does not match number of parameters')
        except:
            if self.npars != 1:
                raise Exception('Number of priors given does not match number of parameters')

             
    
    
    
    names=["minimum","maximum","precision","labels"]    
    check=[self.min,self.max,self.precision,self.labels]

    for x, nam in zip(check,names):
        if np.any(x == None):
             raise Exception('Please set parameter '+str(nam))
        else:
             if x.size != self.npars:
                 raise Exception('Number of constraints in '+str(nam)+' does not match number of parameters')
         
    if np.any(self.fixed == None):
             self.fixed=np.zeros(self.npars, dtype=bool)
             print("You did not specify if any variables are fixed - I will continue assuming that none are")
        

             
    if np.any((self.max-self.min) < 0):
             raise Exception('Parameter(s) '+str(self.labels[(self.max-self.min) < 0])+' have incorrect minumum/maximum bounds')
 
    if np.any((self.guesses<self.min)):
             raise Exception('Parameter(s) '+str(self.labels[(self.guesses<self.min)])+' have an initial guess lower than the minimum allowed.')
             
    if np.any((self.guesses>self.max)):
             raise Exception('Parameter(s) '+str(self.labels[(self.guesses>self.max)])+' have an initial guess higher than the maximum allowed.')


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

    if ((0.8*(niters/self.nprocesses)) < 1000) & (self.nprocesses > 1):
        print("WARNING: The chain assigned to each processor will be very short (<1250 steps) - consider reducing 'nprocesses'.")


    if not numatonce:  numatonce=50*self.npars
    if not burn:  
        self.burn=0.2*(niters/self.nprocesses)
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
            
            
    if not self.silent: 
        print("Best fit:")
        for i in range(0,self.labels.size):
            print("  "+self.labels[i]+":",verybestvalues[i])
        print("Starting final chain")

    if (self.silent==False)&(self.nprocesses>1):
        verboselev=10
    else:
        verboselev=0
    results = []
    par= Parallel(n_jobs= self.nprocesses, verbose=verboselev)
    
    results=par(delayed(unwrap_self)(i) for i in zip([self]*self.nprocesses, [verybestvalues]*self.nprocesses,[int(float(niters)/float(self.nprocesses))]*self.nprocesses,[numatonce]*self.nprocesses,[verybestknob]*self.nprocesses, [False]*self.nprocesses, [True]*self.nprocesses,np.arange(self.nprocesses)))
    results=np.array(results,dtype=object)
    
    par._terminate_backend()
    get_reusable_executor().shutdown(wait=True)
    
    outputvalue= np.concatenate(results[:,0],axis=1)
    outputll= np.concatenate(results[:,1])
    
    if outputll.size < 1: 
        print('WARNING: No accepted models. Perhaps you need to increase the number of steps?')
    else:
        if not self.silent: 
            w,=np.where(outputll == np.max(outputll))
            perc = np.percentile(outputvalue, [15.86, 50, 84.14], axis=1)
            sig_bestfit_up = (perc[2][:] - perc[1][:])
            sig_bestfit_down = (perc[1][:] - perc[0][:])
            
            print("Final best fit values and 1sigma errors:")
            for i in range(0,self.labels.size):
                if self.fixed[i]:
                    print("  "+self.labels[i]+":",perc[1][i],"(Fixed)")
                else:
                    if np.abs(((sig_bestfit_up[i]/sig_bestfit_down[i])-1)) < 0.1:
                        print("  "+self.labels[i]+":",perc[1][i],"±",np.mean([sig_bestfit_up[i],sig_bestfit_down[i]]))
                    else:                    
                        print("  "+self.labels[i]+":",perc[1][i],"+",sig_bestfit_up[i],"-",sig_bestfit_down[i])
    return outputvalue, outputll
