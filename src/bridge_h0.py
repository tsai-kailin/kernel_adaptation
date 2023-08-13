from utils import *
from cme import ConditionalMeanEmbed
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import time
import scipy.sparse as ss


class Bridge_h0:
  """ Construct the bridge function h0 = \sum_i alpha_ij \phi(w_i)\otimes\phi(c_j)
      vec(alpha)=(Gamma_xc\odot I)(n2*lam I + \Sigma)^{-1}y, alpha shape=(n1_samples, n2_samples)
      Gamma_xc = mu_w_cx.get_mean_embed(x,c)['Gamma'] #(n1_samples, n2_samples)
      \Sigma = (Gamma_xc^T K_ww Gamma_xc)K_cc
  """
  def __init__(self, Cw_xc, covars, Y, lam, scale=1.,x0=None, method='original'):
    """Initiate the parameters
    Args:
      Cw_xc: object, ConditionalMeanEmbed
      covars: covariates, dict {"C": ndarray shape=(n2_samples, n1_features), "X": ndarray shape=(n2_samples, n2_features)}
      Y: labels, (n2_samples,)
      lam: reuglarization parameter, lam
      scale: kernel length scale
    """
    t1 = time.time()
    self.sc = scale
    n_sample = Y.shape[0]
    # construct A matrix
    C = covars["C"]

    K_CC = ker_mat(jnp.array(C), jnp.array(C), self.sc)
    self.C = C
    params = Cw_xc.get_params()
    W = params["Y"]
    self.w_sc = params["scale"]
    self.W = W
    K_WW = ker_mat(jnp.array(W), jnp.array(W), params["scale"])

    assert(set(params["Xlist"]) == set(covars.keys()))
    # construct Gamma_xc matrix
    Gamma_xc = Cw_xc.get_mean_embed(covars)["Gamma"] #shape = (n1_samples, n2_samples)
    

    # construct sigma
    Sigma = Hadamard_prod(mat_mul(mat_mul(Gamma_xc.T, K_WW), Gamma_xc), K_CC)



    #plt.plot(jnp.linalg.eigvalsh(mat_mul(D.T,D)))
    #u, vh = jsla.eigh(Sigma)
    #idx = jnp.where(u>1e-6)[0]
    #u2 = 1./(u[idx]+n_sample*lam)
    #F = mat_mul(vh[:,idx]*u2, vh[:,idx].T)


    #print("rank of sigma", jnp.linalg.matrix_rank(Sigma))
    F = Sigma + n_sample*lam*jnp.eye(n_sample)
    #print("F is pd", is_pos_def(F))
    
    t2 = time.time()


    
    #using linear solver
    
    
    if method == 'nystrom':
      print('use Nystrom method to estimate h0')
      #q = min(2*int(np.sqrt(n_sample)), int(n_sample/10))
      q = min(250, n_sample)
      select_id = np.random.choice(n_sample, q, replace=False)
      
      K_q = Sigma[select_id, :][:, select_id]
      #evaluate the spectrum of K
      #v, uh = jnp.linalg.eigh(Sigma)
      #num_id = jnp.where(jnp.abs(v)>1e-5)[0]
      #print("eigenvalue greater than 1e-5: ", num_id.size, n_sample)


      K_nq = Sigma[:, select_id]
      
      #inv_Kq = jsla.solve(K_q+1e-5*jnp.eye(q), jnp.eye(q), assume_a='pos')
      #if jnp.isnan(inv_Kq).any():
      #  print("inv_Kq is nan 1")

      inv_Kq_sqrt =  jnp.array(truncate_sqrtinv(K_q))
      Q = K_nq.dot(inv_Kq_sqrt)

      print('kernel approximation error:', jnp.linalg.norm(Sigma - Q.dot(Q.T)))

      #inversion method 1
      #inv_K = jsla.solve(lam*n_sample*inv_Kq + mat_mul(K_nq.T, K_nq), jnp.eye(q), assume_a='pos')
      #inv_Kq = jsla.solve(lam*n_sample*K_q + mat_mul(K_nq.T, K_nq), jnp.eye(q), assume_a='pos')
      #inversion method 2
      #inv_K = jsla.solve(lam*n_sample*jnp.eye(q) + mat_mul(inv_Kq, mat_mul(K_nq.T, K_nq)) , jnp.eye(q))
      
      #if jnp.isnan(inv_K).any():
      #  print("inv_K is nan 2")
      #inversion method 1
      #aprox_K = (jnp.eye(n_sample)-mat_mul(mat_mul(K_nq, inv_K), K_nq.T))/(lam*n_sample)
      #inversion methos 2
      #aprox_K = (jnp.eye(n_sample)-mat_mul(mat_mul(K_nq, inv_K), mat_mul(inv_Kq, K_nq.T)))/(lam*n_sample)

      inv_temp = jsla.solve(lam*n_sample*jnp.eye(q)+Q.T.dot(Q), jnp.eye(q))
      if jnp.isnan(inv_temp).any():
        print("inv_temp is nan")         
      aprox_K = (jnp.eye(n_sample)-(Q.dot(inv_temp)).dot(Q.T))/(lam*n_sample)

      vec_alpha = aprox_K.dot(Y)
    elif method == 'cg':
      print('use conjugate gradeint to estimate h0')
      #using conjugate gradient descent
      F = F.at[jnp.abs(F)<1e-5].set(0.0)
      sparse_F = ss.csr_matrix(np.array(F))
      vec_alpha, exit_code = ss.linalg.cg(sparse_F, np.array(Y), x0)
      print('solve status:', exit_code)
      vec_alpha = jnp.array(vec_alpha)
    else:
      print('use linear solver to estimate h0')
      vec_alpha = jsla.solve(F, Y)
    
    #use Nystrom approximation method
    """
    q = min(5*int(np.sqrt(n_sample)), int(n_sample/10))
    select_id = np.random.choice(n_sample, q, replace=True)

    K_qC = ker_mat(jnp.array(C[select_id]), jnp.array(select_id), self.sc)
    K_qW_XC = jnp.ones((q, q))
    q_covars = {}
    for key in covars.keys():
      d = covars[key]
      if len(d.shape)>1:
        q_covars[key] = d[select_id,:]
      else:
        q_covars[key] = d[select_id]
    
    Gamma_qxc = Cw_xc.get_mean_embed(q_covars)["Gamma"]
    K_qW_XC = mat_mul(mat_mul(Gamma_qxc.T, K_WW), Gamma_qxc)
    K_q = Hadamard_prod(K_qC, K_qW_XC)
    

    K_nqC = ker_mat(jnp.array(C), jnp.array(select_id), self.sc)
    K_nqW_XC = mat_mul(mat_mul(Gamma_xc.T, K_WW), Gamma_qxc)
    K_nq = Hadamard_prod(K_nqC, K_nqW_XC)

    inv_Kq = jsla.solve(lam*n_sample*jsla.solve(K_q, jnp.eye(q)) + mat_mul(K_nq.T, K_nq), jnp.eye(q))
    aprox_K = (jnp.eye(n_sample)-mat_mul(mat_mul(K_nq, inv_Kq), K_nq.T))/(lam*n_sample)
    vec_alpha = mat_mul(aprox_K, Y)
    """
    t25 = time.time()
    #slower method
    #G = katri_rao_col(Gamma_xc,  jnp.eye(n_sample))
    #vec_alpha = mat_mul(G, vec_alpha)

    #faster method
    
    vec_alpha = stage2_weights(Gamma_xc, vec_alpha)
    t3 = time.time()
    print("processing time: matrix preparation:%.4f solving inverse:%.4f, %.4f"%(t2-t1, t25-t2, t3-t25))
    self.alpha = vec_alpha.reshape((-1, n_sample)) #shape=(n1_sample, n2_sample)


  def __call__(self, new_w, new_c):
    """return h0(w,c)
    Args:
        new_w: variable W, ndarray shape = (n3_samples, n1_features)
        new_c: variable C, ndarray shape = (n3_samples, n2_features)}
    Returns:
        h0(w,c): ndarray shape = (n3_samples)
    """
    # compute K_newWW
    K_WnewW = ker_mat(jnp.array(self.W), jnp.array(new_w), self.w_sc) #(n1_sample, n3_sample)


    # compute K_newCC
    K_CnewC = ker_mat(jnp.array(self.C), jnp.array(new_c), self.sc) #(n2_sample, n3_sample)


    h_wc = fn = lambda kc, kw: jnp.dot(mat_mul(self.alpha, kc), kw)
    v = vmap(h_wc, (1,1))
    return v(K_CnewC, K_WnewW)

  def get_EYx(self, new_x, cme_WC_x):
    """ when computing E[Y|c,x]=<h0, phi(c)\otimes mu_w|x,c>
    Args:
      new_x: ndarray shape=(n4_samples, n_features)
      cme_WC_x: ConditionalMeanEmbed
    """
    #TODO
    t1 = time.time()
    params = cme_WC_x.get_mean_embed(new_x)
    t2 = time.time()
    if len(self.W.shape) == 1:
      w_features = 1
    else:
      w_features = self.W.shape[1]

    if len(self.C.shape) == 1:
      c_features = 1
    else:
      c_features = self.C.shape[1]

    # params["Y"] shape=(n1_samples, w_features+c_features)
    new_w = params["Y"][:, 0:w_features]
    new_c = params["Y"][:, w_features:w_features+c_features]
    # Gamma shape=(n1_samples, n4_samples)
    kcTalphakw = self.__call__(new_w, new_c)
    t3 = time.time()
    fn = lambda x: jnp.dot(kcTalphakw, x)
    v = vmap(fn, (1))

    result = v(params["Gamma"])
    t4 = time.time()

    print("inference time: %.4f/%.4f/%.4f"%(t2-t1, t3-t2, t4-t3))
    return result

  def get_EYx_independent(self, new_x, cme_w_x, cme_c_x):
    """ E[Y | x] = <h0, cme_w_x \otimes cme_c_x>
    Args:
      new_x: ndarray shape=(n5_samples, n_features)
      cme_w_x: ConditionalMeanEmbed, object
      cme_c_x: CME_m0, object
    """
    t1 = time.time()
    #params_w = cme_w_x.get_params()
    new_w = cme_w_x.Y #params_w["Y"]
    Gamma_w = cme_w_x.get_mean_embed(new_x)['Gamma'] #(n3_samples, n5_samples)


    #params_c = cme_c_x.get_params()
    new_c = cme_c_x.C #params_c["C"]

    Gamma_c = cme_c_x.get_A_operator(cme_w_x, new_x)['beta'].T #(n4_sample, n5_sample)
    t2 = time.time()
    # compute K_newWW
    K_WnewW = ker_mat(jnp.array(self.W), jnp.array(new_w), self.w_sc) #(n1_sample, n3_sample)
    # compute K_newCC
    K_CnewC = ker_mat(jnp.array(self.C), jnp.array(new_c), self.sc) #(n2_sample, n4_sample)

    kcTalphakw = mat_mul(K_WnewW.T, mat_mul(self.alpha,K_CnewC)) #(n3_sample,  n4_sample)
    t3 = time.time()

    h_wc = fn = lambda b1, b2: jnp.dot(mat_mul(kcTalphakw, b1), b2)
    v = vmap(h_wc, (1,1))
    result = v(Gamma_c, Gamma_w)
    t4 = time.time()

    print("inference time: %.4f/%.4f/%.4f"%(t2-t1, t3-t2, t4-t3))
    return result

  def get_EYx_independent_cme(self, new_x, cme_w_x, cme_c_x):
    """ E[Y | x] = <h0, cme_w_x \otimes cme_c_x>
    Args:
      new_x: ndarray shape=(n5_samples, n_features)
      cme_w_x: ConditionalMeanEmbed, object
      cme_c_x: ConditionalMeanEmbed, object
    """
    t1 = time.time()
    #params_w = cme_w_x.get_params()
    new_w = cme_w_x.Y #params_w["Y"]
    Gamma_w = cme_w_x.get_mean_embed(new_x)['Gamma'] #(n3_samples, n5_samples)


    #params_c = cme_c_x.get_params()
    new_c = cme_c_x.Y #params_c["Y"]
    Gamma_c = cme_c_x.get_mean_embed(new_x)['Gamma'] #(n4_sample, n5_sample)
    t2 = time.time()
    # compute K_newWW
    K_WnewW = ker_mat(jnp.array(self.W), jnp.array(new_w), self.w_sc) #(n1_sample, n3_sample)
    # compute K_newCC
    K_CnewC = ker_mat(jnp.array(self.C), jnp.array(new_c), self.sc) #(n2_sample, n4_sample)

    kwTalphakc = mat_mul(K_WnewW.T, mat_mul(self.alpha,K_CnewC)) #(n3_sample,  n4_sample)
    t3 = time.time()

    h_wc = fn = lambda b1, b2: jnp.dot(mat_mul(kwTalphakc, b1), b2)
    v = vmap(h_wc, (1,1))
    result = v(Gamma_c, Gamma_w)

    t4 = time.time()

    print("inference time: %.4f/%.4f/%.4f"%(t2-t1, t3-t2, t4-t3))
    return result





