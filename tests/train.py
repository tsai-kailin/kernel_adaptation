import sys
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt


#sys.path.append('../kadapt/')
from kadapt.models.plain_kernel.bridge_h0 import Bridge_h0
from kadapt.models.plain_kernel.bridge_m0 import CME_m0_cme
from kadapt.models.plain_kernel.cme import ConditionalMeanEmbed
from kadapt.gen_data import *
from kadapt.utils import *

import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--cme_m', type=str, default='original',
                    help='conditional mean embed estimation method')
parser.add_argument('--h0_m', type=str, default='original',
                    help='bridge function h0 estimation method')
parser.add_argument('--m0_m', type=str, default='original',
                    help='bridge function m0 estimation method') 

parser.add_argument('--cme_lam',  type=float, default=1e-3,
                    help='lambda for conditional mean embed, set -1 for generalized cross validation')                   
parser.add_argument('--h0_lam',  type=float, default=1e-3,
                    help='lambda for bridge h0')   
parser.add_argument('--m0_lam',  type=float, default=1e-3,
                    help='lambda for bridge m0, set -1 for generalized cross validation') 

parser.add_argument('--task',  type=int, default=1,
                    help='task id: 1 or 2') 


args = parser.parse_args()


#specify esitmation method
cme_method = args.cme_m
h0_method  = args.h0_m #'original'
m0_method  = args.m0_m

lam_cme = args.cme_lam # setting None will activate generalized cross validation
lam_h0 = args.h0_lam
lam_m0 = args.m0_lam

if lam_cme == -1.:
  lam_cme = None

if lam_m0 == -1.:
  lam_m0 = None


scale=1.
task_id = args.task

lam_str = ''
for lam in [lam_cme, lam_h0, lam_m0]:
  if lam is not None:
    lam_str += str(-int(np.log10(lam)))
  else:
    lam_str += 'n'

filename = cme_method[0]+h0_method[0]+m0_method[0]+'_'+lam_str+'_task_'+str(task_id)


if task_id == 1:
  gen_C = gen_C_task1
  gen_Y = gen_Y_task1
elif task_id == 2:
  gen_C = gen_C_task2
  gen_Y = gen_Y_task2

#generate data

# parameters

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
data_list = {}
n = 20000
for sd in sd_lst[:4]:
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
  data_list[sd]=data


#generate data from target domain
def gen_U_target(n,key):
    #e1=random.uniform(key[0],(n,),minval=0,maxval=1)
    """
    e1= -1*random.beta(key[0],1,3,(n,))
    U2= random.beta(key[0],5,1,(n,))
    e3= np.where((U2>1),0,-1)
    e4= np.where((U2<0),0,-1)
    e5=(e3+e4)
    U1=e1 #+e5+1
    """
    #return U1, U2
    

    e1=0.5*random.uniform(key[0],(n,),minval=0,maxval=1)
    U2=(3*random.uniform(key[1],(n,),minval=0,maxval=1)-1)*0.5
    e3= np.where((U2>1),0,-1)
    e4= np.where((U2<0),0,-1)
    e5=(e3+e4)
    U1=e1+e5+1

    return U1, U2
    
target_data_list = {}
n = 20000
for sd in sd_lst[:6]:
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
  target_data_list[sd]=data



# training on the source domain

n_list = [20,50,100,200, 500,700,1000,1500,2000, 2500,3000,4000,5000,7000,10000]#, 12000, 15000]#, 20000, 22000,25000,28000,30000,32000,35000,40000]




data_list.keys()
source_error_list = []
source_error_fapp_list = []
source_estimator_list = []
source_error_fappm0_list = []





for id, n in enumerate(n_list):
  est = {}



  # key=5949, estimate mu^p_{w|x,c}
  # construct covars
  data1 = data_list[5949]
  covars = {}
  covars['X'] = data1['X'][0:n,:]
  covars['C'] = data1['C'][0:n]


  # estimate

  cme_W_XC_p = ConditionalMeanEmbed(data1['W'][0:n,:], covars, lam_cme, scale, method=cme_method)

  Xlist = cme_W_XC_p.get_params()['Xlist']
  covarsx = {}
  covarsx['X'] = data1["X"][0:n,:]

  # concatenate W C
  WC = jnp.hstack((data1['W'][0:n,:], data1['C'][0:n,jnp.newaxis]))
  # estimate mu^p_{wx|c}

  cme_WC_X_p = ConditionalMeanEmbed(WC, covarsx, lam_cme, scale,  method=cme_method)
  # [Partial Identification] estimate mu^p_{w|x}

  cme_W_X_p = ConditionalMeanEmbed(data1['W'][0:n,:], covarsx, lam_cme, scale,  method=cme_method)

  cme_C_X_p = ConditionalMeanEmbed(data1['C'][0:n], covarsx, lam_cme, scale,  method=cme_method)
  # [Partial Identification] key=2807, estimate m0^p
  data4 = data_list[2807]
  covars = {}
  covars['X'] = data4['X'][0:n,:]
  covars['C'] = data4['C'][0:n]

  #m0_p = CME_m0_approx(cme_W_X_p, covars, lam, scale)
  
  m0_p = CME_m0_cme(cme_W_X_p, covars, lam_m0, scale,  method=m0_method)


  # key=7422, estimate h0^p
  data2 = data_list[7422]
  covars2 = {}
  for key in Xlist:
    if len(data2[key].shape)>1:
      covars2[key] = data2[key][0:n,:]
    else:
      covars2[key] = data2[key][0:n]

  h0_p = Bridge_h0(cme_W_XC_p, covars2, data2['Y'][0:n], lam_h0, scale,  method=h0_method)





  #key=4388, testing: compute E[Y|x]
  data3 = data_list[4388]
  testX = {}
  testX["X"] = data3["X"][0:1000,:]
  testY = data3["Y"][0:1000]
  predictY = h0_p.get_EYx(testX, cme_WC_X_p)

  partial_predictY2 = h0_p.get_EYx_independent(testX, cme_W_X_p, m0_p)
  l2_error_appm0 = jnp.sum((testY-partial_predictY2)**2)

  source_error_fappm0_list.append(l2_error_appm0)


  partial_predictY = h0_p.get_EYx_independent_cme(testX, cme_W_X_p, cme_C_X_p)
  l2_error = jnp.sum((testY-predictY)**2)
  source_error_list.append(l2_error)
  print('sample size :', n)
  print("[full] estimation error:", l2_error/predictY.shape[0])
  l2_error_app = jnp.sum((testY-partial_predictY)**2)
  print("[front-door approx] estimation error:", l2_error_app/predictY.shape[0])
  print("[front-door approx] estimation error m0:", l2_error_appm0/predictY.shape[0])
  source_error_fapp_list.append(l2_error_app)
  #store estimator
  est['nsample'] = n
  #est['lam'] = lam
  est['cme_w_xc'] = cme_W_XC_p
  est['cme_w_x']  = cme_W_X_p
  est['cme_wc_x'] = cme_WC_X_p
  est['h0'] = h0_p
  est['m0'] = m0_p

  source_estimator_list.append(est)


# training on the target domain


#Ey|X from target distribution



target_error_list = []
target_error_fapp_list = []
target_error_fappm0_list = []
target_estimator_list = []


for n in n_list:
  # key=2807 estimate mu_{w|x,c}^q)
  # construct covars
  data1 = target_data_list[2807]
  covars = {}
  covars['X'] = data1['X'][0:n,:]
  covars['C'] = data1['C'][0:n]

  cme_W_XC_q = ConditionalMeanEmbed(data1['W'][0:n,:], covars, lam_cme, scale,  method=cme_method)
  Xlist = cme_W_XC_q.get_params()['Xlist']

  covarsx = {}
  covarsx['X'] = data1["X"][0:n,:]
  # concatenate W C
  WC = jnp.hstack((data1['W'][0:n,:], data1['C'][0:n,jnp.newaxis]))
  # estimate mu^p_{wx|c}
  cme_WC_X_q = ConditionalMeanEmbed(WC, covarsx, lam_cme, scale,  method=cme_method)



  # [Partial Identification] estimate mu^p_{w|x}
  cme_W_X_q = ConditionalMeanEmbed(data1['W'][0:n,:], covarsx, lam_cme, scale,  method=cme_method)
  cme_C_X_q = ConditionalMeanEmbed(data1['C'][0:n], covarsx, lam_cme, scale,  method=cme_method)

  # [Partial Identification] key=2807, estimate m0^p
  data4 = target_data_list[2807]
  covars = {}
  covars['X'] = data4['X'][0:n,:]
  covars['C'] = data4['C'][0:n]

  m0_q = CME_m0_cme(cme_W_X_q, covars, lam_m0, scale,  method=m0_method)


  #seed 5654 estimate bridge
  data2 = target_data_list[7422]
  covars2 = {}
  for key in Xlist:
    if len(data2[key].shape)>1:
      covars2[key] = data2[key][0:n,:]
    else:
      covars2[key] = data2[key][0:n]
  h0_q = Bridge_h0(cme_W_XC_q, covars2, data2['Y'][0:n], lam_h0, scale,  method=h0_method)



  #seed 5518 esting: compute E[Y|x]
  data3 = target_data_list[5518]
  testX = {}
  testX["X"] = data3["X"][0:1000,:]
  testY = data3["Y"][0:1000]
  predictY = h0_q.get_EYx(testX, cme_WC_X_q)
  partial_predictY2 = h0_q.get_EYx_independent(testX, cme_W_X_q, m0_q)

  l2_error_appm0 = jnp.sum((testY-partial_predictY2)**2)
  partial_predictY = h0_q.get_EYx_independent_cme(testX, cme_W_X_q, cme_C_X_q)
  l2_error_app = jnp.sum((testY-partial_predictY)**2)

  l2_error = jnp.sum((testY-predictY)**2)
  target_error_list.append(l2_error)
  print("estimation error:", l2_error/predictY.shape[0])
  print("[front-door approx] estimation error:", l2_error_app/predictY.shape[0])
  print("[front-door approx] estimation error m0:", l2_error_appm0/predictY.shape[0])
  target_error_fapp_list.append(l2_error_app)
  target_error_fappm0_list.append(l2_error_appm0)
  est = {}
  est['nsample'] = n
  #est['lam'] = lam
  est['cme_w_xc'] = cme_W_XC_q
  est['cme_w_x']  = cme_W_X_q
  est['cme_c_x']  = cme_C_X_q
  est['cme_wc_x'] = cme_WC_X_q
  est['h0'] = h0_q
  est['m0'] = m0_q

  target_estimator_list.append(est)


#prediction: source on target
sournce2target_error_list = []
print('source on target error')
for n in n_list:
  est = list(filter(lambda est: est['nsample'] == n, source_estimator_list))[0]


  data3 = target_data_list[5518]
  testX = {}
  testX["X"] = data3["X"][0:1000,:]
  testY = data3["Y"][0:1000]
  predictY = est['h0'].get_EYx(testX, est['cme_wc_x'])
  #predictY = est['h0'].get_EYx(testX, cme_WC_X_q)

  l2_error = jnp.sum((testY-predictY)**2)
  sournce2target_error_list.append(l2_error)
  print("n={}, estimation error:".format(n), l2_error/predictY.shape[0])


#prediction: source on target
adaptation_error_list = []
print('full adaptation error')
for n in n_list:
  source_est = list(filter(lambda est: est['nsample'] == n, source_estimator_list))[0]
  target_est = list(filter(lambda est: est['nsample'] == n, target_estimator_list))[0]



  data3 = target_data_list[5518]
  testX = {}
  testX["X"] = data3["X"][0:1000,:]
  testY = data3["Y"][0:1000]
  #predictY = est['h0'].get_EYx(testX, est['cme_wc_x'])
  predictY = source_est['h0'].get_EYx(testX, target_est['cme_wc_x'])

  l2_error = jnp.sum((testY-predictY)**2)
  adaptation_error_list.append(l2_error)
  print("n={}, estimation error:".format(n), l2_error/predictY.shape[0])




#prediction: source on target
adaptation_error_fapp_list = []
print('partial adaptation error')
for n in n_list:
  source_est = list(filter(lambda est: est['nsample'] == n, source_estimator_list))[0]
  target_est = list(filter(lambda est: est['nsample'] == n, target_estimator_list))[0]


  data3 = target_data_list[5518]
  testX = {}
  testX["X"] = data3["X"][0:1000,:]
  testY = data3["Y"][0:1000]

  predictY = source_est['h0'].get_EYx_independent_cme(testX, target_est['cme_w_x'], target_est['cme_c_x'])

  l2_error = jnp.sum((testY-predictY)**2)
  adaptation_error_fapp_list.append(l2_error)
  print("n={}, estimation error:".format(n), l2_error/predictY.shape[0])

#prediction: source on target
adaptation_error_fappm0_list = []
print('partial adaptation m0 error')

for n in n_list:
  source_est = list(filter(lambda est: est['nsample'] == n, source_estimator_list))[0]
  target_est = list(filter(lambda est: est['nsample'] == n, target_estimator_list))[0]



  data3 = target_data_list[5518]
  testX = {}
  testX["X"] = data3["X"][0:1000,:]
  testY = data3["Y"][0:1000]
  #predictY = est['h0'].get_EYx(testX, est['cme_wc_x'])
  predictY = source_est['h0'].get_EYx_independent(testX, target_est['cme_w_x'], source_est['m0'])
  #predictY = source_est['h0'].get_EYx_independent_cme(testX, target_est['cme_w_x'], target_est['cme_c_x'])

  l2_error = jnp.sum((testY-predictY)**2)
  adaptation_error_fappm0_list.append(l2_error)
  print("n={}, estimation error:".format(n), l2_error/predictY.shape[0])


#prediction: source on target
adaptation_error_fappm02_list = []
print('partial adaptation m02 error')
for n in n_list:
  source_est = list(filter(lambda est: est['nsample'] == n, source_estimator_list))[0]
  target_est = list(filter(lambda est: est['nsample'] == n, target_estimator_list))[0]



  data3 = target_data_list[5518]
  testX = {}
  testX["X"] = data3["X"][0:1000,:]
  testY = data3["Y"][0:1000]
  #predictY = est['h0'].get_EYx(testX, est['cme_wc_x'])
  predictY = source_est['h0'].get_EYx_independent(testX, target_est['cme_w_x'], target_est['m0'])
  #predictY = source_est['h0'].get_EYx_independent_cme(testX, target_est['cme_w_x'], target_est['cme_c_x'])

  l2_error = jnp.sum((testY-predictY)**2)
  adaptation_error_fappm02_list.append(l2_error)
  print("n={}, estimation error:".format(n), l2_error/predictY.shape[0])


#prediction: source on target
adaptation_error_fappm03_list = []
print('partial adaptation m03 error')

for n in n_list:
  source_est = list(filter(lambda est: est['nsample'] == n, source_estimator_list))[0]
  target_est = list(filter(lambda est: est['nsample'] == n, target_estimator_list))[0]



  data3 = target_data_list[5518]
  testX = {}
  testX["X"] = data3["X"][0:1000,:]
  testY = data3["Y"][0:1000]
  #predictY = est['h0'].get_EYx(testX, est['cme_wc_x'])
  predictY = target_est['h0'].get_EYx_independent(testX, target_est['cme_w_x'], source_est['m0'])
  #predictY = source_est['h0'].get_EYx_independent_cme(testX, target_est['cme_w_x'], target_est['cme_c_x'])

  l2_error = jnp.sum((testY-predictY)**2)
  adaptation_error_fappm03_list.append(l2_error)
  print("n={}, estimation error:".format(n), l2_error/predictY.shape[0])


plt.plot(n_list, np.array(target_error_list)/predictY.shape[0],label=r'target on target $\langle \hat{h}_0^q, \hat{\mu}_{WC\mid x}^q\rangle$')
plt.plot(n_list, np.array(target_error_fapp_list)/predictY.shape[0],label=r'target on target (front-door approximation)$\langle \hat{h}_0^q, \hat{\mu}_{W\mid x}^q \otimes \hat{\mu}_{C\mid x}^q\rangle$')
plt.plot(n_list, np.array(target_error_fappm0_list)/predictY.shape[0],label=r'target on target (front-door approximation)$\langle \hat{h}_0^q, \hat{\mu}_{W\mid x}^q \otimes \hat{m}_0^q(\hat{\mu}_{W\mid x}^q, x)\rangle$')


plt.plot(n_list, np.array(sournce2target_error_list)/predictY.shape[0], label=r'source on target $\langle \hat{h}_0^p, \hat{\mu}_{WC\mid x}^p\rangle$')
plt.plot(n_list, np.array(adaptation_error_list)/predictY.shape[0],label= r'adaptation $\langle \hat{h}_0^p, \hat{\mu}_{WC\mid x}^q\rangle$')
plt.plot(n_list, np.array(adaptation_error_fapp_list)/predictY.shape[0],label=r'adaptation (front-door approximation)$\langle \hat{h}_0^p, \hat{\mu}_{W\mid x}^q \otimes \hat{\mu}_{C\mid x}^q\rangle$')
plt.plot(n_list, np.array(adaptation_error_fappm0_list)/predictY.shape[0],label=r'adaptation (front-door approximation)$\langle \hat{h}_0^p, \hat{\mu}_{W\mid x}^q \otimes \hat{m}_0^p(\hat{\mu}_{W\mid x}^q, x)\rangle$')
plt.plot(n_list, np.array(adaptation_error_fappm02_list)/predictY.shape[0],'--',label=r'(semi) adaptation $\langle \hat{h}_0^p, \hat{\mu}_{W\mid x}^q \otimes \hat{m}_0^q(\hat{\mu}_{W\mid x}^q, x)\rangle$')
plt.plot(n_list, np.array(adaptation_error_fappm03_list)/predictY.shape[0],'--',label=r'(semi) adaptation $\langle \hat{h}_0^q, \hat{\mu}_{W\mid x}^q \otimes \hat{m}_0^p(\hat{\mu}_{W\mid x}^q, x)\rangle$')



plt.xlabel("sample size")
plt.ylabel(r"$\frac{1}{n}\sum(y_i-g(x_i))^2$")
plt.xscale('log')
plt.yscale('log')
plt.title(r"Estimation error of the target $g(x)=\langle \hat{h}_0, \hat{\mu}_{WC\mid x}\rangle$")
plt.grid()
plt.legend(loc=(1.1,0.3))
plt.savefig('result_{}.png'.format(filename),  bbox_inches='tight')