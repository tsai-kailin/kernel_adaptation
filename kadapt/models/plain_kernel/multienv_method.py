"""
Implementation of the base kernel estimator
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT


import jax.numpy as jnp





class MultiKernelMethod:
    """
    Base estimator for the adaptation
    split_data(), predict(), evaluation(), are implemented by the child class
    """
    def __init__(self, source_train, target_train, source_test, target_test, split, scale=1, lam_set = None, method_set = None, kernel_dict=None):
        """ Initiate parameters
        Args:
            source_train: dictionary, keys: C,W,X,Y
            target_train: dictionary, keys: C, W, X, Y
            source_test:  dictionary, keys: X, Y
            target_test:  dictionary, keys: X, Y
            split: Boolean, split the training dataset or not. If True, the samples are evenly split into groups. 
            Hence, each estimator receive smaller number of training samples.  
            scale: length-scale of the kernel function, default: 1.  
            lam_set: a dictionary of tuning parameter, set None for leave-one-out estimation
            For example, lam_set={'cme': lam1, 'h0': lam2, 'm0': lam3}
            method_set: a dictionary of optimization methods for different estimators, default is 'original'
            kernel_dict: a dictionary of specified kernel functions
        """
        self.source_train = source_train
        self.target_train = target_train
        self.source_test  = source_test
        self.target_test  = target_test
        self.sc = scale
        self.split = split
        self.fitted = False
        
        if lam_set == None:
            lam_set={'cme': None, 'k0': None}
        

        self.lam_set = lam_set

        if method_set == None:
            method_set = {'cme': 'original', 'k0': 'original'}
        self.method_set = method_set

        if kernel_dict == None:
            kernel_dict['cme_w_xz'] = {'X': 'rbf', 'Z': 'rbf', 'Y':'rbf'} #Y is W
            kernel_dict['cme_w_x']  = {'X': 'rbf', 'Y': 'rbf'} # Y is W
            kernel_dict['k0']       = {'X': 'rbf'}
        
        self.kernel_dict = kernel_dict 

    def fit(self, train_target=True):
        #split dataset
        if self.split:
            self.split_data()
            print('complete data split')
        # learn estimators from the source domain
        self.source_estimator =  self._fit_source_domains(self.source_train)

        # learn estimators from the target domain
        if train_target:
            self.target_estimator =  self._fit_source_domains(self.target_train)
            
        else:
            self.target_estimator = self._fit_target_domain(self.target_train)
        
        self.fitted = True

    def split_data(self):
        """Split data."""
        raise NotImplementedError("Implemented in child class.")


    def predict(self):
        """Fits the model to the training data."""
        raise NotImplementedError("Implemented in child class.")

    def evaluation(self):
        """Fits the model to the training data."""
        raise NotImplementedError("Implemented in child class.")


    def score(self, testY, predictY):
        ## Fix shape
        if testY.shape > predictY.shape:
            assert testY.ndim == predictY.ndim + 1 and testY.shape[:-1] == predictY.shape, "unresolveable shape mismatch betweenn testY and predictY"
            predictY = predictY.reshape(testY.shape)
        elif testY.shape < predictY.shape:
            assert testY.ndim + 1 == predictY.ndim and testY.shape == predictY.shape[:-1], "unresolveable shape mismatch betweenn testY and predictY"
            testY = testY.reshape(predictY.shape)
        l2_error =  jnp.sum((testY-predictY)**2)/predictY.shape[0]
        return l2_error
    

