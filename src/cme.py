
from utils import *
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import scipy


def leverage_scores(K, q, lam):
    """
    code adapted from: 
    C. Williams and M. Seeger, "Using the Nystrom method to speed up kernel machines," in Proceedings of the 14th Annual Conference on Neural Information Processing Systems, 2001, no. EPFL-CONF.161322, pp. 682-688.

    A. Alaoui and M. Mahoney, "Fast Randomized Kernel Methods With Statistical Guarantees", arXiv, 2001.

    Compute leverage scores for matrix K and regularization parameter lbd.

    :param K: (``numpy.ndarray``) or of (``Kinterface``). The kernel to be approximated with G.

    :return: (``numpy.ndarray``) a vector of leverage scores to determine a sampling distribution.
    """
    dg = jnp.diag(K) 
    #print(dg)
    pi = dg / np.sum(dg)
    #print(pi, np.sum(pi))
    n = K.shape[0]
    linxs = np.random.choice(range(n), size=q, replace=True)
    C = K[:, linxs]
    W = C[linxs, :]
    B = C.dot(np.real(scipy.linalg.sqrtm(W)))
    BTB = B.T.dot(B)
    BTBi = np.linalg.inv(BTB + n * lam * np.eye(q, q))
    l = np.array([B[i, :].dot(BTBi).dot(B[i, :]) for i in range(n)])
    return l / np.sum(l)

class ConditionalMeanEmbed:
  """function class of conditional mean embedding
    C(Y|X) = Phi_Y(K_XX+lam*n1_samples*I)^{-1}Phi_X
    mu(Y|x) = C(Y|x=x) = Phi_Y(K_XX+lam*n1_samples*I)^{-1}Phi_X(x)
    E[phi(Y,y)|X=x] = <y, mu(Y|x)>

    Example:
    X  = {}
    n1_samples = 50
    X["X1"] = jax.random.normal(key, shape=(n1_samples,))
    X["X2"] = jax.random.normal(key, shape=(n1_samples, 2))
    Y = jax.random.normal(key2, shape=(n1_samples,))
    C_YX = ConditionalMeanEmbed(Y, X, 0.1)

    new_x = {}
    n2_samples = 5
    new_x["X1"] = jax.random.normal(key, shape=(n2_samples,))
    new_x["X2"] = jax.random.normal(key, shape=(n2_samples, 2))
    n3_samples = 20
    new_y = jax.random.normal(key2, shape=(n3_samples,))
    C_YX(new_y, new_x)
  """
  def __init__(self, Y, X, lam, scale=1, method='original', q=None):
    """ initiate the parameters
      Args:
        Y: dependent variables, ndarray shape=(n1_samples, n2_features)
        X: independent varaibles, dict {"Xi": ndarray shape=(n1_samples, n1_features)}
        lam: regularization parameter
        scale: kernel length scale
    """
    self.n_samples = Y.shape[0]
    self.X_list = list(X.keys())
    self.X = X
    self.Y = Y
    assert(lam >= 0.)
    self.lam = lam
    self.sc = scale
    self.method = method
    # construct of gram matrix
    K_XX = jnp.ones((self.n_samples, self.n_samples))
    for key in self.X_list:
      x = X[key]
      temp = ker_mat(jnp.array(x), jnp.array(x), self.sc)

      K_XX= Hadamard_prod(K_XX, temp)
    self.K_XX = K_XX

    #compute Nystrom approximation
    if self.method=='nystrom':
      if q == None:
      
        q = min(250, self.n_samples)
      if q < self.n_samples:
        select_x = np.random.choice(self.n_samples, q, replace=False)
        #select_x = np.arange(q)
        #unselect_x = np.array(list(set(np.arange(self.n_samples)) - set(select_x)))
        #reorder_x = np.concatenate((select_x, unselect_x))
      else:
        select_x = np.arange(self.n_samples)
        #reorder_x = np.arange(self.n_samples)
      """
      
      K_nq = jnp.ones((self.n_samples, q))
      for key in self.X_list:
        x = X[key]
        if len(x.shape) > 1:
          temp = ker_mat(jnp.array(x), jnp.array(x[select_x,:]), self.sc) 
        else:
          temp = ker_mat(jnp.array(x), jnp.array(x[select_x]), self.sc) 
        K_nq = Hadamard_prod(K_nq, temp)

      K_q = jnp.ones((q, q))
      for key in self.X_list:
        x = X[key]
        if len(x.shape) > 1:
          temp = ker_mat(jnp.array(x[select_x,:]), jnp.array(x[select_x,:]), self.sc) 
        else:
          temp = ker_mat(jnp.array(x[select_x]), jnp.array(x[select_x]), self.sc) 
      K_q = Hadamard_prod(K_q, temp)   
      """
      K_q = self.K_XX[select_x, :][:, select_x]
      K_nq = self.K_XX[:, select_x]

      # evaluate the spectrum of the K_XX
      #u,vh = jnp.linalg.eigh(self.K_XX)
      #num = np.where(np.abs(u)>1e-5)[0].size
      #print('ratio of eigenvalues greater than 1e-5:', num/self.n_samples)

      inv_Kq_sqrt =  jnp.array(truncate_sqrtinv(K_q))

      Q = K_nq.dot(inv_Kq_sqrt)

      #print('kernel approximation error:', jnp.linalg.norm(self.K_XX - Q.dot(Q.T)))
      
      #inv_Kq = jnp.linalg.inv(K_q)
      #if jnp.isnan(inv_Kq).any():
      #  print("inv_Kq is nan 1")
      
      #inversion method 1
      #inv_K = jnp.linalg.inv(self.lam*self.n_samples*inv_Kq + K_nq.T.dot(K_nq))#, jnp.eye(q), assume_a='pos')
      #inversion method 2
      #inv_K = jsla.solve(lam*self.n_samples*jnp.eye(q) + mat_mul(inv_Kq, mat_mul(K_nq.T, K_nq)), jnp.eye(q))
    
      #following is unstable
      #inv_K = jsla.solve(lam*self.n_samples*K_q + mat_mul(K_nq.T, K_nq), jnp.eye(q))
      #if jnp.isnan(inv_K).any():
      #  print("inv_K is nan 2")   
      # for inversion method 1
      #self.aprox_K_XX = (jnp.eye(self.n_samples)-mat_mul(K_nq.dot(inv_K), K_nq.T))/(self.lam*self.n_samples)
      
      
      inv_temp = jsla.solve(self.lam*self.n_samples*jnp.eye(q)+Q.T.dot(Q), jnp.eye(q))
      if jnp.isnan(inv_temp).any():
        print("inv_temp is nan")         
      self.aprox_K_XX = (jnp.eye(self.n_samples)-(Q.dot(inv_temp)).dot(Q.T))/(self.lam*self.n_samples)

      # for inversion method 2
      # self.aprox_K_XX = (jnp.eye(self.n_samples)-mat_mul(mat_mul(K_nq, inv_K), mat_mul(inv_Kq, K_nq.T)))/(lam*self.n_samples)      
      
      # evaluation
      Gx = self.K_XX + self.lam*self.n_samples*jnp.eye(self.n_samples)
      inv_Gx = jsla.solve(Gx, jnp.eye(self.n_samples), assume_a='pos')

      #print('distance: ', jnp.linalg.norm(inv_Gx-self.aprox_K_XX))
      
  def lam_selection(self):
    """
    """
    #TODO: implement leave-one-out method to select lambda
    pass
  def get_params(self):
    """Return parameters.
    """
    Gx = self.K_XX + self.lam*self.n_samples*jnp.eye(self.n_samples)

    #K_YY = ker_mat(jnp.array(self.Y), jnp.array(self.Y), self.sc)
    out_dict = {"GramX": Gx, "Y":self.Y, "X":self.X, "Xlist":self.X_list, "scale":self.sc}
    return out_dict
  

  def get_mean_embed(self, new_x):
    """ compute the mean embedding given new_x C(Y|new_x)
      Args:
        new_x: independent varaibles, dict {"Xi": ndarray shape=(n2_samples, n1_features)}
      Returns:
    """
    
    Gx = self.K_XX + self.lam*self.n_samples*jnp.eye(self.n_samples)
    n2_samples = new_x[self.X_list[0]].shape[0]

    Phi_Xnx = jnp.ones((self.n_samples, n2_samples))

    for key in self.X_list:
      temp = ker_mat(jnp.array(self.X[key]), jnp.array(new_x[key]), self.sc)
      Phi_Xnx = Hadamard_prod(Phi_Xnx, temp)

    #Gamma = (K_XX+lam*n1_samples*I)^{-1}Phi_X(new_x)
    #print("GX is pd",is_pos_def(Gx))
    #if self.n_samples < 1000:
      #direct solver
    
    

      #use Nystrom approximation
    if self.method == 'nystrom':
      #print('use Nystrom method to estimate cme')
      Gamma = self.aprox_K_XX.dot(Phi_Xnx)
    elif self.method == 'cg':
      #print('use conjugate gradient to estimate cme')
      #use conjugate gradient descent
      Gx = Gx.at[jnp.abs(Gx)<1e-5].set(0.0)
      fn = lambda x: jax.scipy.sparse.linalg.cg(Gx, x)[0]
      v = vmap(fn, 1)
      Gamma = v(Phi_Xnx).T
    else:
      #print('use linear solver to estimate cme')
      #print(self.X_list)
      #Gamma= jsla.solve(Gx, Phi_Xnx, assume_a='pos') #shape=(n1_samples, n2_samples),
      Gx = self.K_XX + self.lam*self.n_samples*jnp.eye(self.n_samples)
      inv_Gx = jsla.solve(Gx, jnp.eye(self.n_samples), assume_a='pos')
      Gamma = inv_Gx.dot(Phi_Xnx)
    
    evaluate = False
    if evaluate:
      Gx = self.K_XX + self.lam*self.n_samples*jnp.eye(self.n_samples)
      inv_Gx = jsla.solve(Gx, jnp.eye(self.n_samples), assume_a='pos')
      Gamma2 = inv_Gx.dot(Phi_Xnx)
      print('difference of Gamma', jnp.linalg.norm(Gamma-Gamma2))
      
    
    return {"Y": self.Y, "Gamma": Gamma, "scale": self.sc} # jnp.dot(kernel(Y,y; sc), Gamma)

  def __call__(self, new_y, new_x):
    """
      Args:
        new_y: dependent variables, ndarray shape=(n3_samples, n2_features)
        new_x: independent varaibles, dict {"Xi": ndarray shape=(n2_samples, n1_features)}
      Returns:
        out: ndarray shape=(n3_samples, n2_samples)
    """
    memb_nx = self.get_mean_embed(new_x)
    Gamma = memb_nx["Gamma"]
    Phi_Yny = ker_mat(jnp.array(new_y), jnp.array(self.Y), self.sc)


    return mat_mul(Phi_Yny, Gamma)

  def get_coefs(self, new_x):

    memb_nx = self.get_mean_embed(new_x)
    Gamma = memb_nx["Gamma"]
    return Gamma
