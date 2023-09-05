import sys
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../src/')
from bridge_h0 import Bridge_h0
from bridge_m0 import CME_m0_cme
from cme import ConditionalMeanEmbed
from gen_data import *
from utils import *
from adaptation import full_adapt, partial_adapt

####################
# generate data    #
####################


def data_to_df(data, n):
    """ convert dictionary of data to pd.DataFrame
    Args:
        data: dict
        n: number of samples, int
    """

    d_list = [None]*n
    for i in range(n):
        d = {}
        d['U'] = data['U'][i,:]
        d['W'] = data['W'][i,:]
        d['X'] = data['X'][i,:]
        d['Y'] = data['Y'][i]
        d['C'] = data['C'][i]

        d_list[i]=d
    return pd.DataFrame(d_list)
    

m_e=[4,0,0,0,0]
m_u=[0,0,0]
v_u=[1,1,1]
m_x=[0,0]
v_x=[3,1]
m_w=[0,0]
v_w=[1,3]

seed_list = {}
sd_lst= [5949, 7422, 4388, 2807, 5654, 5518, 1816, 1102, 9886, 1656, 4379,
       2029, 8455, 4987, 4259, 2533, 9783, 7987, 1009, 2297]


#set number of samples to train
n = 8000
source_data_list = {}
for sd in sd_lst[:2]:
  seed=[]
  seed1=sd+5446
  seed2=sd+3569
  seed3=sd+10
  seed4=sd+1572
  seed5=sd+42980
  seed6=sd+368641
  seed=[seed1,seed2,seed3,seed4,seed5,seed6]
  num_var=10
  seed_list[sd]=seed

  keyu = random.PRNGKey(seed1)
  keyu, *subkeysu = random.split(keyu, 4)

  keyx = random.PRNGKey(seed2)
  keyx, *subkeysx = random.split(keyx, 4)


  keyw = random.PRNGKey(seed3)
  keyw, *subkeysw = random.split(keyw, 4)

  keyc = random.PRNGKey(seed4)
  keyc, *subkeysa = random.split(keyc, 4)
  U = ((jnp.asarray(gen_U(n,key=subkeysu))).T)
  X = ((jnp.asarray(gen_X(U[:,0],U[:,1], m_x, v_x, n, key=subkeysx))).T)
  W = ((jnp.asarray(gen_W(U[:,0],U[:,1], m_w, v_w, n, key=subkeysw))).T)
  C= (gen_C(U[:,0],U[:,1], X[:,0], X[:,1],0.05, n, key=keyc))
  Y =(gen_Y(C, U[:,0], U[:,1], n))

  # Standardised sample
  Us = standardise(U) [0]
  Xs = standardise(X) [0]
  Ws = standardise(W) [0]
  Cs = standardise(C) [0]
  Ys = standardise(Y) [0]

  data = {}
  data['U'] = U
  data['X'] = X
  data['W'] = W
  data['C'] = C
  data['Y'] = Y
  source_data_list[sd] = data


source_train = source_data_list[5949]

#test set only has 1000 samples
source_test = source_data_list[7422]

#generate data from target domain
def gen_U_target(n,key):

    e1=0.5*random.uniform(key[0],(n,),minval=0,maxval=1)
    U2=(3*random.uniform(key[1],(n,),minval=0,maxval=1)-1)*0.5
    e3= np.where((U2>1),0,-1)
    e4= np.where((U2<0),0,-1)
    e5=(e3+e4)
    U1=e1+e5+1

    return U1, U2


target_data_list = {}

for sd in sd_lst[:2]:
  seed=[]
  seed1=sd+5446
  seed2=sd+3569
  seed3=sd+10
  seed4=sd+1572
  seed5=sd+42980
  seed6=sd+368641
  seed=[seed1,seed2,seed3,seed4,seed5,seed6]
  num_var=10
  seed_list[sd]=seed

  keyu = random.PRNGKey(seed1)
  keyu, *subkeysu = random.split(keyu, 4)

  keyx = random.PRNGKey(seed2)
  keyx, *subkeysx = random.split(keyx, 4)


  keyw = random.PRNGKey(seed3)
  keyw, *subkeysw = random.split(keyw, 4)

  keyc = random.PRNGKey(seed4)
  keyc, *subkeysa = random.split(keyc, 4)
  U = ((jnp.asarray(gen_U_target(n,key=subkeysu))).T)
  X = ((jnp.asarray(gen_X(U[:,0],U[:,1], m_x, v_x, n, key=subkeysx))).T)
  W = ((jnp.asarray(gen_W(U[:,0],U[:,1], m_w, v_w, n, key=subkeysw))).T)
  C= (gen_C(U[:,0],U[:,1], X[:,0], X[:,1],0.05, n, key=keyc))
  Y =(gen_Y(C, U[:,0], U[:,1], n))

  # Standardised sample
  Us = standardise(U) [0]
  Xs = standardise(X) [0]
  Ws = standardise(W) [0]
  Cs = standardise(C) [0]
  Ys = standardise(Y) [0]

  data = {}
  data['U'] = U
  data['X'] = X
  data['W'] = W
  data['C'] = C
  data['Y'] = Y
  target_data_list[sd] = data



target_train = target_data_list[5949]

target_test = target_data_list[7422]

print('data generation complete')

print('source_train number of samples: ', source_train['X'].shape[0])
print('source_test  number of samples: ', source_test['X'].shape[0])
print('target_train number of samples: ', target_train['X'].shape[0])
print('target_test  number of samples: ', target_test['X'].shape[0])



####################
# training...      #
####################
#set the parameter

lam_set = {'cme': 1e-3, 'h0': 1e-3, 'm0': 1e-3}
method_set = {'cme': 'original', 'h0': 'original', 'm0': 'original'}
split = True
scale = 1
estimator_full = full_adapt(source_train, target_train, source_test, target_test, split, scale, lam_set, method_set)

estimator_full.fit()
estimator_full.evaluation()


estimator_partial = partial_adapt(source_train, target_train, source_test, target_test, split, scale, lam_set, method_set)

estimator_partial.fit()
estimator_partial.evaluation()

