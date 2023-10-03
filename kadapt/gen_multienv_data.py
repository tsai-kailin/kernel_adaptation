#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

from jax import random
import numpy as np


def gen_Z(n, mean, sigma, key):
    Z = (random.normal(key,(n,))*sigma)+mean

    return Z




def gen_U(Z, n,key):
    e1=random.uniform(key[0],(n,),minval=0,maxval=1)
    U2=3*random.uniform(key[1],(n,),minval=0,maxval=1)-1
    e3= np.where((Z>0.5),0,-1)
    e4= np.where((Z<-0.5),0,-1)
    e5=(e3+e4)
    U1=e1+e5+1

    return U1, U2


def gen_X(U1,U2, m_x, v_x, n,  key):
    X1= U1+ (random.normal(key[0],(n,))*v_x[0])+m_x[0]
    X2= U2+ random.uniform(key[1],(n,),minval=-1,maxval=1)

    return X1, X2


def gen_W(U1,U2, m_w, v_w, n,  key):
    W1= U1+ random.uniform(key[0],(n,),minval=-1,maxval=1)
    W2= U2+ (random.normal(key[1],(n,)) *v_w[1])+m_w[1]

    return W1, W2


def gen_Y(X1, X2, U1, U2, n):
  mask =  (U1*U2 > 0)
  d = mask.astype(int)*2-1

  y= U2*(np.cos(2*(X1*X2+.3*U1+.2))+d)
  return y
