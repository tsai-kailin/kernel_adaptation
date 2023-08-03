
from utils import *
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsla





class ConditonalMeanEmbed:
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
    C_YX = ConditonalMeanEmbed(Y, X, 0.1)

    new_x = {}
    n2_samples = 5
    new_x["X1"] = jax.random.normal(key, shape=(n2_samples,))
    new_x["X2"] = jax.random.normal(key, shape=(n2_samples, 2))
    n3_samples = 20
    new_y = jax.random.normal(key2, shape=(n3_samples,))
    C_YX(new_y, new_x)
  """
  def __init__(self, Y, X, lam, scale=1, q=None):
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

    # construct of gram matrix
    K_XX = jnp.ones((self.n_samples, self.n_samples))
    for key in self.X_list:
      x = X[key]
      temp = ker_mat(jnp.array(x), jnp.array(x), self.sc)

      K_XX= Hadamard_prod(K_XX, temp)
    self.K_XX = K_XX

    #compute Nystrom approximation
    if q == None:
      q = min(5*int(np.sqrt(self.n_samples)), int(self.n_samples/10))
    select_x = np.random.choice(self.n_samples, q, replace=True)
    K_q = K_XX[select_x, :][:, select_x]
    K_nq = K_XX[:, select_x]
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

    inv_Kq = jsla.solve(K_q, jnp.eye(q))#, assume_a='pos')
    inv_Kq = jsla.solve(lam*self.n_samples*inv_Kq + mat_mul(K_nq.T, K_nq), jnp.eye(q))#, assume_a='pos')
    self.aprox_K_XX = (jnp.eye(self.n_samples)-mat_mul(mat_mul(K_nq, inv_Kq), K_nq.T))/(lam*self.n_samples)


  def lam_selection(self):
    """
    """
    #TODO: implement leave-one-out method to select lambda
    pass
  def get_params(self):
    """Return parameters.
    """
    Gx = self.K_XX + self.lam*self.n_samples*jnp.eye(self.n_samples)

    K_YY = ker_mat(jnp.array(self.Y), jnp.array(self.Y), self.sc)
    out_dict = {"GramX": Gx, "GramY": K_YY, "Y":self.Y, "X":self.X, "Xlist":self.X_list, "scale":self.sc}
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
    #Gamma= jsla.solve(Gx, Phi_Xnx, assume_a='pos') #shape=(n1_samples, n2_samples),
    #else:
      #use conjugate gradient descent
      #Gx = Gx.at[jnp.abs(Gx)<1e-5].set(0.0)
      #fn = lambda x: jax.scipy.sparse.linalg.cg(Gx, x)[0]
      #v = vmap(fn, 1)
      #Gamma = v(Phi_Xnx).T

    #use Nystrom approximation

    Gamma = mat_mul(self.aprox_K_XX, Phi_Xnx)
      

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
