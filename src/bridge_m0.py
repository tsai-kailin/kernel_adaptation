from utils import *
from cme import ConditionalMeanEmbed
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import time
import scipy.sparse as ss
import scipy



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
      Cw_x: object, ConditionalMeanEmbed
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
  def __init__(self, Cw_x, covars, lam, scale=1., method='original'):
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
    
    

    self.X = covars['X']

    #linear solver
   
    K_xx = ker_mat(jnp.array(self.X), jnp.array(self.X), self.sc)

    D = Hadamard_prod(K_xx, kx_g_kx)
    
    if method == 'simple':
        alpha = 1./(jnp.diag(D)+lam*self.n_samples) #very unstable
    elif method == 'cg':
      K_cc = ker_mat(jnp.array(self.C), jnp.array(self.C), self.sc)
      M = Hadamard_prod(Hadamard_prod(K_xx, K_cc), kx_g_kx)
      DC = Hadamard_prod(mat_mul(D,D),K_cc)
      # nystrom M
      q = min(250, self.n_samples)
      select_x = np.random.choice(self.n_samples, q, replace=False)
      K_q = M[select_x, :][:, select_x]
      K_nq = M[:, select_x]

      inv_Kq_sqrt = truncate_sqrtinv(K_q)
      Q = K_nq.dot(inv_Kq_sqrt)
      
      aprox_M = Q.dot(Q.T)
      print('truncate error M', jnp.linalg.norm(M-aprox_M))
      
      # nystrom M^{-1/2}GM^{-1/2}
      inv_M_sqrt = jnp.array(truncate_sqrtinv(aprox_M))
      u, vh = jnp.linalg.eigh(inv_M_sqrt)
      id = np.where(np.abs(u)>1e-5)[0]
      print('eigenvalue grater than 1e-5 inv_M_sqrt:',id.size/self.n_samples)    

      M_sqrt = jnp.array(truncate_sqrt(aprox_M))
      m1 = M_sqrt.sum(axis=0)
      MGM = mat_mul(inv_M_sqrt, DC.dot(inv_M_sqrt))

      Sigma = MGM + lam*self.n_samples*jnp.eye(self.n_samples)
      Sigma = Sigma.at[jnp.abs(Sigma)<1e-5].set(0.0)
      sparse_Sigma = ss.csr_matrix(np.array(Sigma))
      temp_alpha, exit_code = ss.linalg.cg(sparse_Sigma, m1, maxiter=5000)
      print('solve status:', exit_code)


      temp_alpha = inv_M_sqrt.dot(jnp.array(temp_alpha))
      alpha = temp_alpha/(lam*self.n_samples)
      """
        K_cc = ker_mat(jnp.array(self.C), jnp.array(self.C), self.sc)
        M = Hadamard_prod(Hadamard_prod(K_xx, K_cc), kx_g_kx)
        
        DC = Hadamard_prod(mat_mul(D,D),K_cc)
        Sigma = DC+lam*self.n_samples*M
        Sigma = Sigma.at[jnp.abs(Sigma)<1e-5].set(0.0)
        sparse_Sigma = ss.csr_matrix(np.array(Sigma))
        m1 = M.sum(axis=0)
        alpha, exit_code = ss.linalg.cg(sparse_Sigma, np.array(m1), maxiter=5000)
        
        print('solve status:', exit_code)
        alpha = jnp.array(alpha)
      """
    elif method == 'original':
        K_cc = ker_mat(jnp.array(self.C), jnp.array(self.C), self.sc)
        M = Hadamard_prod(Hadamard_prod(K_xx, K_cc), kx_g_kx)
        
        DC = Hadamard_prod(mat_mul(D,D),K_cc)

        #u, vh = jnp.linalg.eigh(DC)
        #idx = np.where(np.abs(u)>1e-5)[0].size
        #print('m0 ratio of eigv > 1e-5:', idx/self.n_samples)

        Sigma = DC+lam*self.n_samples*M

        #u, vh = jnp.linalg.eigh(Sigma)
        #idx = np.where(np.abs(u)>1e-5)[0].size
        #print('sigma ratio of eigv > 1e-5:', idx/self.n_samples)
        m1 = M.sum(axis=0)
        alpha = jsla.solve(Sigma, m1)

    elif method == "nystrom":
      
      K_cc = ker_mat(jnp.array(self.C), jnp.array(self.C), self.sc)
      M = Hadamard_prod(Hadamard_prod(K_xx, K_cc), kx_g_kx)
      DC = Hadamard_prod(D.dot(D), K_cc)

      #test rank

      #u, vh = jnp.linalg.eigh(D)
      #id = np.where(np.abs(u)>1e-5)[0]
      #print('eigenvalue grater than 1e-5 D:',id.size/self.n_samples)    

      #u, vh = jnp.linalg.eigh(D)
      #id = np.where(np.abs(u**2)>1e-5)[0]
      #print('eigenvalue grater than 1e-5 DD(reconstruct):',id.size/self.n_samples)    
      
      #new_u = u[id]
      #new_vh = vh[:, id]
      #DD2 = (new_vh*(new_u**2)).dot(new_vh.T)
      #DC2 = Hadamard_prod(DD2, K_cc)
      #print('approx error:', jnp.linalg.norm(DD2-D.dot(D)))

      #u, vh = jnp.linalg.eigh(D.dot(D))
      #id = np.where(np.abs(u)>1e-5)[0]
      #print('eigenvalue grater than 1e-5 DD:',id.size/self.n_samples)    

      #u, vh = jnp.linalg.eigh(mat_mul(D,D))
      #id = np.where(np.abs(u)>1e-5)[0]
      #print('eigenvalue grater than 1e-5 DD2:',id.size/self.n_samples)    

      #u, vh = jnp.linalg.eigh(K_cc)
      #id = np.where(np.abs(u)>1e-5)[0]
      #print('eigenvalue grater than 1e-5 K_cc:',id.size/self.n_samples)    

      #u, vh = jnp.linalg.eigh(DC)
      #id = np.where(np.abs(u)>1e-5)[0]
      #print('eigenvalue grater than 1e-5 DC:',id.size/self.n_samples)    


      # nystrom M
      q = min(500, self.n_samples)
      select_x = np.random.choice(self.n_samples, q, replace=False)
      K_q = M[select_x, :][:, select_x]
      K_nq = M[:, select_x]

      inv_Kq_sqrt = truncate_sqrtinv(K_q)
      Q = K_nq.dot(inv_Kq_sqrt)
      
      aprox_M = Q.dot(Q.T)
      #print('truncate error M', jnp.linalg.norm(M-aprox_M))
      
      # nystrom M^{-1/2}GM^{-1/2}
      inv_M_sqrt = jnp.array(truncate_sqrtinv(aprox_M))
      #u, vh = jnp.linalg.eigh(inv_M_sqrt)
      #id = np.where(np.abs(u)>1e-5)[0]
      #print('eigenvalue grater than 1e-5 inv_M_sqrt:',id.size/self.n_samples)    

      M_sqrt = jnp.array(truncate_sqrt(aprox_M))

      MGM = inv_M_sqrt.dot(DC.dot(inv_M_sqrt))
      #u, vh = jnp.linalg.eigh(MGM)
      #id = np.where(u>1e-5)[0]
      #print('eigenvalue grater than 1e-5 MGM:',id.size/self.n_samples)
      
      
      q = min(1000, self.n_samples)
      
      select_x2 = np.random.choice(self.n_samples, q, replace=False)
      K_q2 = MGM[select_x2, :][:, select_x2]
      K_nq2 = MGM[:, select_x2]
      
      inv_Kq2_sqrt = truncate_sqrtinv(K_q2)
      Q2 = K_nq2.dot(inv_Kq2_sqrt)
      
      aprox_inv = woodbury_identity(Q2, lam, self.n_samples)

      temp_alpha = inv_M_sqrt.dot(aprox_inv.dot(M_sqrt.sum(axis=1)))
      alpha = temp_alpha/(lam*self.n_samples)


    elif method == "nystrom2":
      K_cc = ker_mat(jnp.array(self.C), jnp.array(self.C), self.sc)
      M = Hadamard_prod(D, K_cc)
      
      u, vh = jnp.linalg.eigh(D)
      id = np.where(np.abs(u**2)>1e-5)[0]
      print('eigenvalue grater than 1e-5 DD(reconstruct):',id.size/self.n_samples)    
      
      new_u = u[id]
      new_vh = vh[:, id]
      DD2 = (new_vh*(new_u**2)).dot(new_vh.T)
      DC = Hadamard_prod(D.dot(D), K_cc)
      DC2 = Hadamard_prod(DD2, K_cc)

      # compute the nystrom for M
      q = min(500, self.n_samples)
      select_x = np.random.choice(self.n_samples, q, replace=False)
      K_q = M[select_x, :][:, select_x]
      K_nq = M[:, select_x]
      inv_Kq_sqrt = truncate_sqrtinv(K_q)
      Q = K_nq.dot(inv_Kq_sqrt)
      aprox_M = Q.dot(Q.T)

      aprox_invM = truncate_inv(aprox_M)
      print('approximation error for M:', jnp.linalg.norm(M-aprox_M))

      u, vh = jnp.linalg.eigh(M)
      select_id = np.where(u>1e-5)[0]
      new_u = u[select_id]
      new_vh = vh[:, select_id]
      invM = truncate_inv(M)
      #invM = jsla.solve(M, jnp.eye(self.n_samples))
      print('approximation error for invM:', jnp.linalg.norm(invM-aprox_invM))
      
      # compute the nystrom for DC2
      q = min(1000, self.n_samples)
      select_x = np.random.choice(self.n_samples, q, replace=False)    
      K_q2 = DC2[select_x, :][:, select_x]
      K_nq2 = DC2[:, select_x]
      inv_Kq2_sqrt = truncate_sqrtinv(K_q2)
      Q2 = K_nq2.dot(inv_Kq2_sqrt)
      #print('approximation error for DC:', jnp.linalg.norm(DC2-Q2.dot(Q2.T)))
      """
      inv_temp = jsla.solve(lam*self.n_samples*jnp.eye(q)+Q2.T.dot(Q2), jnp.eye(q))
      temp = Q2.dot(inv_temp.dot(Q2.T))
      alpha = (jnp.eye(self.n_samples)- invM.dot(temp)).sum(axis=0)/(lam*self.n_samples)
      """
      aprox_DC = Q2.dot(Q2.T) 
      print('aproximation error for DC', jnp.linalg.norm(DC-aprox_DC))
      Sigma = truncate_inv(M).dot(aprox_DC)+lam*self.n_samples*jnp.eye(self.n_samples)
      
      
      Sigma2 = DC2 + lam*self.n_samples*M
      print('aproximation error for Sigma', jnp.linalg.norm(Sigma2-Sigma))

      m1 = M.sum(axis=0)

      #alpha = jsla.solve(Sigma2, m1)
      alpha = jsla.solve(Sigma, jnp.ones(self.n_samples))
    #Conjugate gradient descent    

    elif method == "gradient":
      #implementation of gradient descent.
      K_cc = ker_mat(jnp.array(self.C), jnp.array(self.C), self.sc)

      M = Hadamard_prod(D, K_cc)
      max_itr = 5000
      thre = 1e-3
      res = thre*3
      itr = 0
      alpha = jnp.zeros(self.n_samples)
      step = 1e-1
      m1 = M.sum(axis=0)
      
      DC = Hadamard_prod(mat_mul(D,D.T),K_cc)
      Sigma = DC+lam*self.n_samples*M

      while(res>thre and itr < max_itr):
      #for _ in range(1000):
        itr += 1
        new_alpha = alpha - 2*step*(-m1+Sigma.dot(alpha))/(self.n_samples)
        res = jnp.linalg.norm(2*step*(-m1+Sigma.dot(alpha))/(self.n_samples))
        alpha = new_alpha
      print('itr:', itr)

    #simplified version
    
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





class CME_m0_cme:
  """ Construct conditonal mean embedding that embeds the bridge function m0.
  Double conditional mean embedding.
  """
  def __init__(self, Cw_x, covars, lam, scale=1., q=None, method='original'):
    """
    Args:
      Cw_x: ConditionalMeanEmbed, object
      covars: dictionary of covariates, dict
      lam: tuning parametier, float
      scale: kernel length-scale, float
      q: rank of the matrix, when Nystrom approximation is used, int
      method: method, "original" or "nystrom"
    """
    self.method = method
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

    self.X = covars['X']
   
    K_xx = ker_mat(jnp.array(self.X), jnp.array(self.X), self.sc)
    #build the kernel matrix
    self.K_gram =  Hadamard_prod(K_xx, kx_g_kx)
    self.C = covars['C']
    self.n_samples = self.C.shape[0]
    self.lam = lam


    if self.method=='nystrom':
      # set rank
      if q == None:
        q = min(250, self.n_samples)
      
      # set selected indices
      if q < self.n_samples: 
        select_x = np.random.choice(self.n_samples, q, replace=False)
      else:
        select_x = np.aranges(self.n_samples)

      K_q = self.K_gram[select_x, :][:, select_x]
      K_nq = self.K_gram[:, select_x]

      inv_Kq_sqrt = jnp.array(truncate_sqrtinv(K_q))
      Q = mat_mul(K_nq, inv_Kq_sqrt)


      inv_temp = jsla.solve(self.lam*self.n_samples*jnp.eye(q)+Q.T.dot(Q), jnp.eye(q))
      if jnp.isnan(inv_temp).any():
        print("inv_temp is nan")         
      self.aprox_K_gram_inv = (jnp.eye(self.n_samples)-(Q.dot(inv_temp)).dot(Q.T))/(self.lam*self.n_samples)

    elif self.method=='original':
      self.K_gram_inv = jsla.solve(self.lam*self.n_samples*jnp.eye(self.n_samples)+self.K_gram, jnp.eye(self.n_samples))


  def get_mean_embed(self, Cw_x, new_x):
    """
    Args:
      Cw_x: ConditionalMeanEmbed, object
      new_x: shape (n2_samples, n2_features)
    """
    
    # compute the gram matrix
    K_Xnewx = ker_mat(jnp.array(self.X), jnp.array(new_x['X']), self.sc)
    
    
    params1 = Cw_x.get_mean_embed(new_x)
    Gamma1_newx = params1["Gamma"] #(n_samples, n6_samples)
    W1 = params1["Y"]

    K_w1w2 = ker_mat(jnp.array(self.W), jnp.array(W1), self.w_sc) #(n_samples, n'_samples)
    kx_g_knewx = mat_mul(mat_mul(self.Gamma_x.T, K_w1w2), Gamma1_newx) #(n5_samples, n6_samples)

    G_x = Hadamard_prod(K_Xnewx, kx_g_knewx)

    if self.method == 'nystrom':
      Gamma = mat_mul(self.aprox_K_gram_inv, G_x)
    elif self.method == 'original':
      Gamma = mat_mul(self.K_gram_inv, G_x)

    return Gamma



  def get_A_operator(self, Cw_x, new_x):
    """ return \sum_i beta_i(new_x)\phi(c_i)
    Args:
      Cw_x: ConditionalMeanEmbed object
      new_x: shape (n2_samples, n2_features)
    Returns:
    beta: shape (n2_samples, n_samples)
    """
    beta = self.get_mean_embed(Cw_x, new_x).T
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
