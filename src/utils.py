import os,sys
import time
import numpy as np
import pandas as pd
import functools
from typing import Callable
import jax.scipy.linalg as jsla
import jax.scipy.sparse.linalg as jsla_sparse
import jax.numpy.linalg as jnla
import operator
from typing import Dict, Any, Iterator, Tuple

import scipy as sp
import scipy.sparse as sps
import scipy.linalg as la
from numpy.linalg import matrix_rank


from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

import numba
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit, vmap
from jax import random


# utility functions
# clone from: https://github.com/yuchen-zhu/kernel_proxies/blob/main/KPV/utils.py


@jax.jit
def modist(v):
    return jnp.median(v)

@jax.jit
def sum_jit(A,B):
    return jnp.sum(A,B)

@jax.jit
def linear_kern(x, y):
    return jnp.sum(x * y)

@jax.jit
def l2_dist(x,y):
    return jnp.array((x - y)**2)

#@functools.partial(jax.jit, static_argnums=(0,1))
def identifier(x,y):
    if (x!=y):
        b=0
    else:
        b=1
    return b


@functools.partial(jax.jit, static_argnums=(0))
def dist_func(func1: Callable, x,y):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func1( x1, y1))(y))(x)


@jax.jit
def rbf_ker(x,y,scale=1):
    dist_mat=dist_func(l2_dist,x,y)
    gamma=modist(jnp.sqrt(dist_mat))
    #gamma=1
    #coef=1/(2*gamma**2)
    coef=1/(2*scale*(gamma**2))
    return jnp.exp(-coef*dist_mat)

@jax.jit
def identifier_ker(x,y):
    return dist_func(identifier,x,y)


@jax.jit
def Hadamard_prod(A,B):
    return A*B


@jax.jit
def jsla_inv(A):
    return jsla.inv(A)

@jax.jit
def jnla_norm(A):
  return jnla.norm(A)

@jax.jit
def kron_prod(a,b):
    return jnp.kron(a,b)


@jax.jit
def modif_kron(x,y):
    if (y.shape[1]!=x.shape[1]):
        print("Column_number error")
    else:
        return jnp.array(list(jnp.kron(x[:,i], y[:,i]).T for i in list(range(y.shape[1]))))

@jax.jit
def mat_mul(A,B):
    return jnp.matmul(A,B)

@jax.jit
def jsla_solve(A,B):
    return jax.sp.linalg.solve(A, B, assume_a = 'pos')


@jax.jit
def katri_rao_col(a,b):
  fn = lambda x,y: kron_prod(x,y)
  v = vmap(fn, (1,1),1)
  return v(a,b)


def integral_rbf_ker(x,y, ori_scale):
  """
  compute new gram matrix such that each entry is \tilde{K}(x,y)=\int K(z,x)K(z,y)dz, where K is the original kernel function

  """
  dist_mat=dist_func(l2_dist,x,y)
  gamma=modist(jnp.sqrt(dist_mat))
  new_l = ori_scale*2
  new_gram = rbf_ker(x,y,new_l)*jnp.sqrt(jnp.pi*ori_scale)*gamma
  return new_gram

def ker_mat(X1,X2, scale):
  """
  compute the K_xx
  """
  K_x1x2 = rbf_ker(jnp.array(X1), jnp.array(X2), scale)
  if len(K_x1x2.shape) == 3:
    #perform Hadmard product
    K_x1x2 = jnp.prod(K_x1x2, axis=2)
  return K_x1x2


def stage2_weights(Gamma_w, Sigma_inv):
            n_row = Gamma_w.shape[0]
            arr = [mat_mul(jnp.diag(Gamma_w[i, :]), Sigma_inv) for i in range(n_row)]
            return jnp.concatenate(arr, axis=0)


def standardise(X):
    scaler = StandardScaler()
    if X.ndim == 1:
        X_scaled = scaler.fit_transform(X.reshape(-1,1)).squeeze()
        return X_scaled, scaler
    else:
        X_scaled = scaler.fit_transform(X).squeeze()
        return X_scaled, scaler