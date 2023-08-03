from utils import *
from cme import ConditonalMeanEmbed
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import time





class CME_m0:
  """ construct conditonal mean embedding that embeds the bridge function m0.
  A = \sum_ij alpha_ij phi(c_j)\otimes phi(w_i)\otimes phi(x_j)
  alpha: ndarray, shape=(n4_samples, n5_samples)

  alpha = (\Sigma+lam*n5*I)^{-1}(I \katri-rao-c (K_xx+lam*n4I)^{-1}K_xx)\one
  G = (I \katri-rao-c (K_xx+lam*n4I)^{-1}K_xx)
  \Sigma = (K_xc^{-1} \kron K_ww^{-1})(K_cc\kron \one)(Kx \katri-rao-c K_ww(K_xx+lam*n4I)^{-1}K_xx)(Kx \katri-rao-c K_ww(K_xx+lam*n4I)^{-1}K_xx)^T
  """
  def __init__(self, Cw_x, covars, lam, scale=1.):
    """Initiate the parameters
    Args:
      Cw_x: object, ConditonalMeanEmbed
      covars: covariates, dict {"Xi": ndarray shape=(n5_samples, n1_features)}
      : labels, (n2_samples,)
      lam: reuglarization parameter, lam
      scale: kernel length scale
    """
    self.sc = scale



    # construct \Lambda^T=K_w(K_XX+lambda I)^{-1}K_x

    params = Cw_x.get_params()
    self.W = params['Y']

    self.X = covars['X']
    K_xx = ker_mat(jnp.array(self.X), jnp.array(self.X), self.sc)

    self.C = covars['C']
    self.n_samples = self.C.shape[0]
    K_cc = ker_mat(jnp.array(self.C), jnp.array(self.C), self.sc)

    covarsx = {}
    covarsx['X'] = covars['X']



    Gamma_x = Cw_x.get_mean_embed(covarsx)["Gamma"] #shape = (n1_samples, n2_samples)
    G = katri_rao_col(jnp.eye(self.n_samples), Gamma_x)

    vec_y = jnp.sum(G, axis=1)


    # construct Sigma
    K_xc = Hadamard_prod(K_xx, K_cc)
    K_xc_inv = jsla.solve(K_xc, jnp.eye(self.n_samples))
    self.w_sc = params['scale']
    K_ww = ker_mat(jnp.array(self.W), jnp.array(self.W), params['scale'])
    K_ww_inv = jsla.solve(K_ww, jnp.eye(K_ww.shape[0]))
    Omega = kron_prod(K_xc_inv, K_ww_inv)
    # construct Psi
    Psi = kron_prod(K_cc, jnp.eye(K_ww.shape[0]))

    Lambda = katri_rao_col(K_xx, Gamma_x)

    Sigma = mat_mul(Omega, Hadamard_prod(Psi, mat_mul(Lambda, Lambda.T)))
    Sigma = Sigma + lam*self.n_samples*jnp.eye(Sigma.shape[0])
    #Sigma = Hadamard_prod(Psi, mat_mul(Lambda, Lambda.T))+lam*self.n_samples*kron_prod(K_xc, K_ww)
    u, _ = jnp.linalg.eigh(Sigma)
    #print('eigen value', u)

    vec_alpha = jsla.solve(Sigma, vec_y)

    self.alpha = vec_alpha.reshape((-1, self.n_samples)) #shape=(n4_sample, n5_sample)


  def get_params(self):
    params = {}
    params['X'] = self.X
    params['W'] = self.W
    params['C'] = self.C
    params['scale'] = self.sc
    params['w_scale'] = self.w_sc
    params['alpha'] = self.alpha
    return params

  def get_A_operator(self, Cw_x, new_x):
    """ Get thee A operator
    A[Cw_x, new_x] = Kx(new_x)^T(K_xx+lam I)^{-1}Kw2w\alpha(Kx(new_x)\odot Kc(cdot))
                   = \sum \beta_i(new_x) phi(c_i)
    Notation:
      Gamma: Kx(new_x)^T(K_xx+lam I)^{-1}K_ww
      K_newxX: Kx(new_x)

    Args:
      Cw_x: ConditionalMeanEmbed
      new_x: ndarray shape=(n6_samples, n_features)
    """
    params = Cw_x.get_mean_embed(new_x)

    W2 = params['Y']
    K_w2W = ker_mat(jnp.array(W2), jnp.array(self.W), params['scale'])
    Gamma = mat_mul(mat_mul(params['Gamma'].T, K_w2W), self.alpha) # shape=(n6_samples, n5_sanples)
    X2 = new_x['X']
    K_newxX = ker_mat(jnp.array(X2), jnp.array(self.X), self.sc)  # shape=(n6_samples, n5_sanples)
    beta = Hadamard_prod(Gamma, K_newxX)
    out_dict = {"Gamma": Gamma, "K_newxX": K_newxX, "C": self.C, "beta":beta, "scale":self.sc}
    return out_dict

  def __call__(self, new_c, Cw_x, new_x):
    """Compute E(k(new_c,C)|new_x), shape (n7_samples, n6_samples)
    Args:
      new_c: ndarray shape=(n7_samples, n_features)
      Cw_x: ConditionalMeanEmbed
      new_x: ndarray shape=(n6_samples, n_features)
    """
    params = self.get_A_operator(Cw_x, new_x)
    #K_newcC = ker_mat(jnp.array(new_c), jnp.array(params["C"]), params["scale"]) # shape=(n7_samples, n5_samples)
    K_Cnewc = ker_mat(jnp.array(params["C"]), jnp.array(new_c), params["scale"])
    return mat_mul(params['beta'], K_Cnewc).T

    #temp = vmap(lambda x1: vmap(lambda y1: x1*y1)(params['K_newxX']))(K_newcC) # shape=(n7_samples, n6_samples, n5_samples)
    #return vmap(lambda y1: vmap(lambda x,y: jnp.dot(x,y))(params['Gamma'],y1))(temp)


class CME_m0_ver2(CME_m0):
  """ construct conditonal mean embedding that embeds the bridge function m0.
  A = \sum_ij alpha_i phi(c_i)\otimes cme_w_x(x_i) \otimes phi(x_i)
  alpha: ndarray, shape=(n5_samples)

  alpha = (lam*n5*M+E)^{-1}M1
  M = K_c\odot K_x\odot K_xx(K_xx+lam*n4*I)^{-1}K_WW(K_xx+lam*n4*I)^{-1}K_xx
  E = \sum_j diag(d_j)K_CC\diag(d_j)
  d_j = Kx(x_j)\odot K_xx(K_xx+lam*n4*I)^{-1}K_WW(K_xx+lam*n4*I)^{-1}K_x(x_j)
  """
  def __init__(self, Cw_x, covars, lam, scale=1.):
    self.sc = scale
    self.Cw_x = Cw_x
    params = Cw_x.get_params()
    self.W = params['Y']
    self.w_sc = params['scale']
    covarsx = {}
    covarsx['X'] = covars['X']
    K_ww = ker_mat(jnp.array(self.W), jnp.array(self.W), params['scale'])
    self.Gamma_x = Cw_x.get_mean_embed(covarsx)["Gamma"]

    kx_g_kx = mat_mul(self.Gamma_x.T, mat_mul(K_ww, self.Gamma_x))

    self.C = covars['C']
    self.n_samples = self.C.shape[0]
    K_cc = ker_mat(jnp.array(self.C), jnp.array(self.C), self.sc)


    self.X = covars['X']
    K_xx = ker_mat(jnp.array(self.X), jnp.array(self.X), self.sc)

    M = Hadamard_prod(Hadamard_prod(K_xx, K_cc), kx_g_kx)
    D = Hadamard_prod(K_xx, kx_g_kx)
    DC = Hadamard_prod(mat_mul(D,D),K_cc)

    fn = lambda x: x*K_cc*x
    v = vmap(fn, (1))
    #print("rank of M", jnp.linalg.matrix_rank(M))
    #print("sparsity", (M>=1e-5).sum()/M.size)
    #U = v(D).sum(axis=0)
    #print("rank of v(D)", jnp.linalg.matrix_rank(U))
    #print("sparsity", (U>=1e-5).sum()/U.size)



    #Sigma = DC+lam*self.n_samples*M
    #m1 = M.sum(axis=0)
    #alpha = jsla.solve(Sigma, m1)
    
    #Sigma = Sigma.at[jnp.abs(Sigma)<1e-5].set(0.0)
    #sparse_Sigma = ss.csr_matrix(np.array(Sigma))
    #alpha, exit_code = scipy.sparse.linalg.cg(sparse_Sigma, np.array(m1), maxiter=5000)
    
    #print('solve status:', exit_code)
    #alpha2 = jnp.array(alpha)
    
    alpha = 1./(jnp.diag(D)+lam*self.n_samples)

    #print('error: ', jnp.linalg.norm(alpha-alpha2), jnp.linalg.norm(alpha-alpha3), jnp.linalg.norm(alpha2-alpha3))
    #Sigma = jsla.solve(M,v(D).sum(axis=0))+lam*self.n_samples*jnp.eye(self.n_samples)
    #alpha = jsla.solve(Sigma, jnp.ones(self.n_samples))
    self.alpha = alpha

  def get_A_operator(self, Cw_x, new_x):
    """ return \sum_i beta_i(new_x)\phi(c_i)
    Args:
      Cw_x: ConditionalMeanEmbed object
      new_x: shape (n6_samples, n2_features)
    Returns:
    beta: shape (n6_samples, n5_samples)
    """
    K_Xnewx = ker_mat(jnp.array(self.X), jnp.array(new_x['X']), self.sc) #(n5_samples, n6_samples)
    params1 = Cw_x.get_mean_embed(new_x)
    Gamma1_newx = params1["Gamma"] #(n_samples, n6_samples)
    W1 = params1["Y"]

    K_w1w2 = ker_mat(jnp.array(self.W), jnp.array(W1), self.w_sc) #(n_samples, n'_samples)
    B = mat_mul(mat_mul(self.Gamma_x.T, K_w1w2), Gamma1_newx) #(n5_samples, n6_samples)
    #beta = (Hadamard_prod(B, K_Xnewx).T)*self.alpha #(n6_samples, n5_samples)
    fn = lambda x: Hadamard_prod(x, self.alpha)
    v = vmap(fn)
    beta = v(Hadamard_prod(B, K_Xnewx).T)
    params = {}
    params["C"]=self.C
    params["scale"] = self.sc
    params["beta"] = beta
    return params


  def __call__(self, new_c, Cw_x, new_x):

    params = self.get_A_operator(Cw_x, new_x)
    K_Cnewc = ker_mat(jnp.array(self.C), jnp.array(new_c), self.w_sc) #(n5_samples, n2_samples)
    return Hadamard_prod(params['beta'].T, K_Cnewc).sum(axis=0)

  def get_coefs(self, Cw_x, new_x):
    params = self.get_A_operator(Cw_x, new_x)
    return params['beta'].T #(n5_samples, n2_samples)



