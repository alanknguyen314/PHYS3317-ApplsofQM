
# coding: utf-8

# # PHYS 3317 -- Angular Momentum

# In[1]:


from __future__ import division


# In[2]:


from scipy.linalg import eigh


# In[84]:


from pylab import *


# In[3]:


from IPython.display import Math
from IPython.display import Latex
from IPython.display import display
import IPython.display
import types


# In[73]:


class BasisState():
    """ state=BasisState(name,spin,mz) reprents the state of a spin (or other angular momentum object)
    with name name, total spin "spin", and spin projection mz.  You can access these attributes with:
    state.name, state.spin, and state.mz

    Example:
    L0=BasisState("L",1,0)
    display(L0)
    print(L0.spin)
    print(L0.name)
    print(L0.mz)

    S1=BasisState("S",1/2,1/2)
    display(S12)
    P=L0*S12 # make product state
    display(P)

    S0=BasisState("S",1/2,-1/2)
    display(S0)

    psi=(S0+S1)/sqrt(2) # make wavefunction which is a linear superposition
    """
    
    def __init__(self,name,spin,mz) :
        self.name=name
        self.spin=spin
        self.mz=mz
        self.isstate=True
        
    def _repr_latex_(self):
        return r"$|\,"+str(self.name)+"="+str(self.spin)+";m_{"+str(self.name)+"}="+str(self.mz)+"\\rangle$"
    
    def copy(self):
        return(BasisState(self.name,self.spin,self.mz))
    
    def __getitem__(self,name):
        if self.name==name:
            return self
        elif self==name:
            return 1
        else :
            return 0
        
    def __neg__(self):
        return self*(-1)
        
    def __rmul__(self,other):
        return self*other
    
    def __div__(self,other):
        return self*(1./other)

    def __truediv__(self,other):
        return self*(1./other)
    
    def __mul__(self,other):
        try: # are we multiplying by a productstate?
            return ProductState(self,*other.basisstates)
        except:
            try: #are we multiplying by a basisstate?
                return ProductState(self,other)
            except: #are we multiplying by something else?
                return wavefunction([1],[self])*other
                
    def __add__(self,other):
        return wavefunction([1],[self])+other
    
    def __eq__(self,other):
        try:
            samename=self.name==other.name
            samespin=self.spin==other.spin
            samesz=self.mz==other.mz
            return samename and samespin and samesz
        except:
            return False
    
    def overlap(self,other):
        return wavefunction([1],[self]).overlap(other)
    
    def __ne__(self,other):
        return not(self==other)
    
    def __hash__(self):
        namehash = self.name.__hash__()
        spinhash= self.spin.__hash__()
        szhash=self.mz.__hash__()
        return namehash^spinhash^szhash


# In[85]:


class ProductState():
    """ProductState represents a product of any number of spins -- for example if s1 and s2 are two BasisState objects
    ProductState(s1,s2) represents the outer product.  Not usually directly called by the user-- equivalent to: 
    s1*s2"""

    def __init__(self,*states):
        self.name=[]
        self.spin=[]
        self.mz=[]
        for state in states:
            self.name.append(state.name)
            self.spin.append(state.spin)
            self.mz.append(state.mz)
        self.basisstates=states
        self.isstate=True

    def index(self,name):
        """ProductState.index(name) returns the index corresponding to that name"""
        return self.name.index(name)

    def __contains__(self,name):
        try :
            return name in self.basisstates
        except :
            return name in self.name

    def __getitem__(self,name):
        if name in self.name:
            ind = self.index(name)
            return self.basisstates[ind]
        elif name in self.basisstates:
            return 1
        else:
            return 0
    
    def __neg__(self):
        return self*(-1)
    
    def __rmul__(self,other):
        return self*other
        
    def __mul__(self,other):
        try:
            allstates=self.basisstates+other.basisstates
            return ProductState(*allstates)
        except:
            try:
                return ProductState(other,*self.basisstates)
            except:
                return wavefunction([1],[self])*other
    
    def __truediv__(self,other):
        return self*(1./other)
    
    def __div__(self,other):
        return self*(1./other)
    
    def __add__(self,other):
        return wavefunction([1],[self])+other
        
    def __iter__(self):
        return self.basisstates.__iter__()

    def getmz(self,name):
        ind = self.index(name)
        return self.mz[ind]

    def setmz(self,name,mzval):
        ind=self.index(name)
        self.mz[ind]=mzval
        self.basisstates[ind].mz=mzval

    def copy(self):
        new = ProductState()
        new.name=self.name[:]
        new.spin=self.spin[:]
        new.mz=self.mz[:]
        new.basisstates=[st.copy() for st in self.basisstates]
        return new
        
    def _repr_latex_(self):
        rstring=r"$"
        for state in self:
            tmpstring=state._repr_latex_()
            rstring=rstring+tmpstring[1:-1]+"\\otimes"
        rstring=rstring[:-7]+"$"
        return rstring
    
    def overlap(self,other):
        return wavefunction([1],[self]).overlap(other)
    
    def __eq__(self,other):
        names1=set(self.basisstates)
        names2=set(other.basisstates)
        return names1==names2
    
    def __ne__(self,other):
        return not(self==other)
    
    def __hash__(self):
        h=0
        for state in self:
            h=h^(state.__hash__())
        return h
        


# In[86]:


class wavefunction():
    """ A wavefunction(coefficients,states) represents the linear combination of the states in states, with 
    weights given by coefficients"""
    
    def __init__(self,coefficients=None,states=None):
        #self.coefficients=array(coefficients)
        #self.states=states
        self.statedic=dict(zip(states,coefficients))
        
    #def index(self,state):
    #    return self.states.index(state)
    
    def __iter__(self):
        #return self.states.__iter__()
        return self.statedic.__iter__()
    
    def __contains__(self,state):
        #return state in self.states
        return state in self.statedic
        
    def __getitem__(self,state):
        if state in self:
            return self.statedic[state]
        else:
            return 0
       # try : # is something sent which can be iterated over
       #     return [self[st] for st in state]
       # except :
       #     try : # check to see if an integer is sent -- interpret it as index of state vector
       #         return self.coefficients[state]
       #     except : # otherwise assume that a state is being checked
       #         if state in self.states:
       #             return self.coefficients[self.index(state)]
       #         else:
       #             return 0

    def __neg__(self):
        return (self*(-1))
    
    def __add__(self,wf2):
        if hasattr(wf2,"statedic"):
            newdic=self.statedic.copy()
            for state in wf2:
                if state in self :
                    newcoef=newdic[state]+wf2[state]
                    if abs(newcoef)<10e-5:
                        newdic.pop(state)
                    else:
                        newdic[state]=newdic[state]+wf2[state]
                else :
                    newdic[state]=wf2[state]
            states=newdic.keys()
            coefs = [newdic[state] for state in states]
            return wavefunction(coefs,states)
        else:
            return self+wavefunction([1],[wf2])
        
        
    def __mul__(self,num):
        if hasattr(num,"statedic"): # worst case scenario -- it is a wavefunction
            statelist=[]
            coeflist=[]
            for state1 in self:
                 for state2 in num:
                    newstate=state1*state2
                    newcoef=self[state1]*num[state2]
                    statelist.append(newstate)
                    coeflist.append(newcoef)
            return wavefunction(coeflist,statelist)
        elif hasattr(num,"isstate"): # it is just a state
            statelist=[]
            coeflist=[]
            for state1 in self:
                newstate=state1*num
                coeflist.appen(self[state1])
                statelist.append(newstate)
            return wavefunction(coeflist,statelist)
        else: #num is just a number
            statelist=[]
            coeflist=[]
            for state in self.statedic:
                statelist.append(state)
                coeflist.append(num*self.statedic[state])
            return wavefunction(coeflist,statelist)
    
    def __sub__(self,w2):
        return self+((-1)*w2)
    
    def __truediv__(self,val):
        return self*(1./val)
  
    
    def __rmul__(self,num):
        return self*num
    
    def dot(self,wf2):
        if not(hasattr(wf2,"statedic")):
            wf=wavefunction([1],[wf2])
        else:
            wf=wf2
        prod=0.
        for st in self:
            if st in wf:
                prod=prod+conjugate(self[st])*wf[st]
        return prod
    
    def overlap(self,wf2):
        return self.dot(wf2)
    
    def norm(self):
        return sqrt(self.dot(self))
    
    def _repr_latex_(self):
        if len(self.statedic)==0:
            return r"$0$"
        else:
            rstring=r"$"
            for state,coef in self.statedic.items():
                rstring=rstring+str(coef)+"\,"+state._repr_latex_()[1:-1]+"\,+\,"
            rstring=rstring[:-5]+"$"
            return rstring
        


# In[12]:


def isorepr(self):
    if self.mz>0:
        iso="u"
    else:
        iso="d"
    return r"$|\,"+str(self.name)+"=\,"+iso+"\\rangle$"


# In[13]:


class isospin(BasisState):
    
    def _repr_latex_(self):
        if self.mz>0:
            iso="u"
        else:
            iso="d"
        #return r"$|\,"+str(self.name)+"=\,"+iso+"\\rangle$"
        return r"$("+iso+")$"


# In[14]:


class antiisospin(BasisState):
    
    def _repr_latex_(self):
        if self.mz>0:
            iso="-\\bar{d}"
        else:
            iso="\\bar{u}"
        #return r"$|\,"+str(self.name)+"=\,"+iso+"\\rangle$"
        return r"$("+iso+")$"


# In[98]:


import types


# In[421]:


class spinoperator():
    """ An abstract class used to define an operator that works on spin indices"""
    
    def __init__(self):
        self.isop=True
        self.reprstring=""
        pass
    
    def dot(self,vec):
        return self(vec) 
    
    def __truediv__(self,n):
        return self*(1./n)
    
    def __rmul__(self,op):
        newop=spinoperator()
        def sm(new,target):
            return op*self(target)
        newop._call=types.MethodType(sm,newop)  # monkey patch the new method in
        newop.reprstring=r"$"+str(op)+"*"+self.reprstring[1:]
        return newop
    
    def __mul__(self,op):
        try:
            return self(op)
        except:
            newop=spinoperator()
            def sm(new,target):
                return self(target)*op
            newop._call=types.MethodType(sm,newop)  # monkey patch the new method in
            newop.reprstring=self.reprstring[:-1]+"*"+str(op)+"$"
            return newop
    
    def __sub__(self,op):
        newop=spinoperator()  # create a new object which will be the operator that represents the sum
        def sm(new,target):   # define the new method
            return self(target)-op(target)
        newop._call=types.MethodType(sm,newop)  # monkey patch the new method in 
        newop.reprstring=r"$("+self.reprstring[1:-1]+"-"+op.reprstring[1:-1]+")$"
        return newop
        # as an aside, the other way to do this (avoiding monkey patching) would be to
        # dynamically define a new class, and instantiate it -- might be prettier
    
    def __add__(self,op):
        newop=spinoperator()  # create a new object which will be the operator that represents the sum
        def sm(new,target):   # define the new method
            return self(target)+op(target)
        newop._call=types.MethodType(sm,newop)  # monkey patch the new method in
        newop.reprstring=r"$("+self.reprstring[1:-1]+"+"+op.reprstring[1:-1]+")$"
        return newop
    
    def __call__(self,op):
        return self._call(op)
            
    def _call(self,op):  
        if hasattr(op,"isop"):
            return self.opcall(op)
        else:
            return self.veccall(op)
            
    def opcall(self,op):
        newop=spinoperator()
        def chain(new,target):
            return self(op(target))
        newop._call=types.MethodType(chain,newop)
        newop.reprstring=self.reprstring[:-1]+op.reprstring[1:]
        return newop
    
    def veccall(self,vec):
        raise Exception('Action of operator on states/numbers has not yet been set')
    
    def _repr_latex_(self):
        return self.reprstring

    
    


# In[173]:


class Sz(spinoperator):
    """An operator which corresponds to the z-component of spin"""
    
    def __init__(self,name):
        self.name=name
        self.reprstring=r"${\bf "+str(self.name)+"}_{z}$"
        self.isop=True
        
    def veccall(self,vec):
        #states=vec.states
        #coefs=vec.coefficients #pointer to coefficients
        finalstates=[]
        finalcoefs=[]
        if not(hasattr(vec,"statedic")):
            vec=wavefunction([1],[vec])
        for v in vec: #loop over the (product) basis states in the vector
            bstate=v[self.name] # get the part of the product involving this operator
            newcoef=bstate.mz*vec[v]
            if abs(newcoef)>10e-5:
                finalstates.append(v)
                finalcoefs.append(newcoef)
        return wavefunction(finalcoefs,finalstates)
    

            
        


# In[174]:


class Splus(spinoperator):
    """An operator which corresponds to the spin raising operator"""
    
    def __init__(self,name):
        self.name=name
        self.reprstring=r"${\bf "+str(self.name)+"}_{+}$"
        self.isop=True
        
    def veccall(self,vec):
        #states=vec.states
        #coefs=vec.coefficients #pointer to coefficients
        finalstates=[]
        finalcoefs=[]
        if hasattr(vec,"statedic"):
            for v in vec: #loop over the (product) basis states in the vector
                newv=v.copy()
                bstate=newv[self.name] # get the part of the product involving this operator
                s=bstate.spin
                mz=bstate.mz
                newcoef=sqrt(s*(s+1)-mz*(mz+1))*vec[v]
                if abs(newcoef)>10e-5:
                    finalcoefs.append(newcoef)
                    bstate.mz=bstate.mz+1
                    finalstates.append(newv)
            return wavefunction(finalcoefs,finalstates)
        else:
            return self.veccall(wavefunction([1],[vec]))


# In[175]:


class Sminus(spinoperator):
    """An operator which corresponds to the spin lowering operator"""
    
    def __init__(self,name):
        self.name=name
        self.reprstring=r"${\bf "+str(self.name)+"}_{-}$"
        self.isop=True
        
    def veccall(self,vec):
        #states=vec.states
        #coefs=vec.coefficients #pointer to coefficients
        finalstates=[]
        finalcoefs=[]
        if hasattr(vec,"statedic"):
            for v in vec: #loop over the (product) basis states in the vector
                newv=v.copy()
                bstate=newv[self.name] # get the part of the product involving this operator
                s=bstate.spin
                mz=bstate.mz
                newcoef=sqrt(s*(s+1)-mz*(mz-1))*vec[v]
                if abs(newcoef)>10e-5:
                    finalcoefs.append(newcoef)
                    bstate.mz=bstate.mz-1
                    finalstates.append(newv)
            return wavefunction(finalcoefs,finalstates)
        else:
            return self.veccall(wavefunction([1],[vec]))


# In[382]:


def matrixrep(operator,basis):
    return array([[operator(state1).overlap(state2) for state2 in basis] for state1 in basis])


# In[199]:


def spinbasis(name,s):
    return [BasisState(name,s,m) for m in arange(-s,s+1,1)]


# In[222]:


def productbasis(*b):
    states=b[0]
    for axis in b[1:]:
        newstates=[]
        for state in axis:
            for oldstate in states:
                newstates.append(state*oldstate)
        states=newstates
    return states

