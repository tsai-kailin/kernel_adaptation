"""
A simple example of the two-stage kernel estimator for multiple environments
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

import sys
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import pandas as pd

#sys.path.append('../kadapt/')

from kadapt.gen_multienv_data import *
from kadapt.utils import *
from kadapt.models.plain_kernel.multienv_adaptation import multienv_adapt_categorical

####################
# generate data    #
####################

n_env = 4
    

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
n = 6000

mean_z =  [-0.5, 0., 0.5, 1.]
sigma_z = [  1., 1.,  1., 1.]

prob_list = [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]]


def gen_batch_data(n_env, n, prob_list, sd_list):
    data_list = []
    for env_id, sd in enumerate(sd_lst[:n_env]):
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

        keyz = random.PRNGKey(seed4)
        keyz, *subkeysz = random.split(keyz, 4)
        Z = ((jnp.asarray(gen_Zcategorical(n, prob_list[env_id], key=keyz))))
        U = ((jnp.asarray(gen_Ucategorical(Z, n, key=subkeysu))).T)
        X = ((jnp.asarray(gen_X(U[:,0],U[:,1], m_x, v_x, n, key=subkeysx))).T)
        W = ((jnp.asarray(gen_W(U[:,0],U[:,1], m_w, v_w, n, key=subkeysw))).T)
        Y = (gen_Y(X[:, 0], X[:, 1], U[:,0], U[:,1], n))

        # Standardised sample
        Us = standardise(U) [0]
        Xs = standardise(X) [0]
        Ws = standardise(W) [0]
        Zs = standardise(Z) [0]
        Ys = standardise(Y) [0]

        data = {}
        data['U'] = U
        data['X'] = X
        data['W'] = W
        data['Z'] = Z
        data['Y'] = Y
        data_list.append(data)

    return data_list


sd_train_list = sd_lst[:n_env]

source_train = gen_batch_data(n_env, n, prob_list, sd_train_list)

#test set only has 1000 samples
sd_test_list = sd_lst[n_env:n_env*2]
source_test = gen_batch_data(n_env, 1000, prob_list, sd_test_list)

#generate data from target domain


target_prob_list = [[0.25, 0.15, 0.15, 0.45]]

target_train = gen_batch_data(1,    n, target_prob_list, [5949])
target_test  = gen_batch_data(1, 1000, target_prob_list, [5654])


print('data generation complete')
print('number of source environments:', len(source_train))
print('source_train number of samples: ', source_train[0]['X'].shape[0]*n_env)
print('source_test  number of samples: ', source_test[0]['X'].shape[0])
print('number of target environments:', len(target_train))
print('target_train number of samples: ', target_train[0]['X'].shape[0])
print('target_test  number of samples: ', target_test[0]['X'].shape[0])



####################
# training...      #
####################
#set the parameter

# set None to use leave-one-out cross-validation method. 
# be aware of the sample size for h0 when using leave-one-out cross-validation, may run into out of memory issue
lam_set = {'cme': 1e-3, 'k0': 1e-3}
method_set = {'cme': 'original', 'k0': 'original'}

#specity the kernel functions for each estimator
kernel_dict = {}

kernel_dict['cme_w_xz'] = {'X': 'rbf', 'Y':'rbf'} #Y is W
kernel_dict['cme_w_x']  = {'X': 'rbf', 'Y': 'rbf'} # Y is W
kernel_dict['k0']       = {'X': 'rbf'}

#splitting the traning data or not, if True the training data will be split evenly into 3 for full-adaptation, and 4 for partial-adaptation
split = True
scale = 1
estimator = multienv_adapt_categorical(source_train, target_train, source_test, target_test, split, scale, lam_set, method_set, kernel_dict)

estimator.fit()
estimator.evaluation()




