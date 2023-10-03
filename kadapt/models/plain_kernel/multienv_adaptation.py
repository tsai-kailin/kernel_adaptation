import pandas as pd
import numpy as np
import jax.numpy as jnp
from kadapt.models.plain_kernel.method import split_data_widx
from kadapt.models.plain_kernel.multienv_method import MultiKernelMethod

from kadapt.models.plain_kernel.cme import ConditionalMeanEmbed
from kadapt.models.plain_kernel.bridge_k0 import Bridge_k0

def concatenate_data(new_data, prev_data):
    """
    new_data: data from new environment
    prev_data: dictionary of data
    """
    
    keys = prev_data.keys()
    concate_data = {}
    for key in keys:
        if prev_data[key] is None:
            concate_data[key] = new_data[key]
        else:
            concate_data[key] = jnp.concatenate((prev_data[key], new_data[key]))

    return concate_data

class multienv_adapt(MultiKernelMethod):
    """
    Adaptation setting: observe (W,X,Y,Z) from multiple environments, (W,X) from the target

    """
    def split_data(self):
        #split training data
        
        n_env = len(self.source_train)
        #print('number of environments of the source:', n_env)
        train_list = [None, None, []]

        #concatenate data from multiple environments
        for i in range(n_env):
            n = self.source_train[i]['X'].shape[0]        
            index = np.random.RandomState(seed=42).permutation(n)
            split_id = np.split(index, [int(n/3), int(n*2/3)])

            if i == 0:
                for j, idx in enumerate(split_id):
                    if j < 2:
                        train_list[j] = split_data_widx(self.source_train[i], idx)
                        
                    else:
                        train_list[2].append(split_data_widx(self.source_train[i], idx))

            else:
                for j, idx in enumerate(split_id):
                    if j < 2:
                        train_list[j] = concatenate_data(split_data_widx(self.source_train[i], idx), train_list[j])
                        
                    else:
                        train_list[2].append(split_data_widx(self.source_train[i], idx))
        
        self.source_train = train_list


        n2 = self.target_train[0]['X'].shape[0]   
        index = np.random.RandomState(seed=42).permutation(n2)
        split_id = np.split(index, [int(n/3), int(n*2/3)])     
        train_list = []
        for j, idx in enumerate(split_id):
            if j == 2:
                train_list.append([split_data_widx(self.target_train[0], idx)])
            else:
                train_list.append(split_data_widx(self.target_train[0], idx))
            
        



        self.target_train = train_list
        


    def _fit_source_domains(self, domain_data):
        """ fit single domain.
        Args:
            domain_data: data to train, [list of data of each environment]
        """



        if self.split:
            train_data = domain_data[0]
        else:
            keys = domain_data.keys()
            empty_dict = dict(zip(keys, [None]*len(keys)))
            train_data = concatenate_data(domain_data, empty_dict) 
        
        covars = {}
        covars['X'] = jnp.array(train_data['X'])
        covars['Z'] = jnp.array(train_data['Z'])

        cme_W_XZ = ConditionalMeanEmbed(jnp.array(train_data['W']), covars, self.lam_set['cme'], 
                                        kernel_dict=self.kernel_dict['cme_w_xz'], scale=self.sc, 
                                        method=self.method_set['cme'])


        # estimate k0
        Xlist = cme_W_XZ.get_params()['Xlist']
        if self.split:
            train_data = domain_data[1]
        else:
            keys = domain_data.keys()
            empty_dict = dict(zip(keys, [None]*len(keys)))
            train_data = concatenate_data(domain_data, empty_dict) 
        
        covars = {}
        for key in Xlist:
            covars[key] = train_data[key]
        k0 = Bridge_k0(cme_W_XZ, covars, train_data['Y'], self.lam_set['k0'], 
                        kernel_dict = self.kernel_dict['k0'], scale = self.sc,  
                        method=self.method_set['k0'])
        
        #esitmate cme_w_x

        if self.split:
            train_data = domain_data[2]
        else:
            train_data = domain_data
        

        estimator = {}
        estimator['cme_w_xz'] = cme_W_XZ
        estimator['cme_w_x'] = {}
        estimator['k0'] = k0

        #estimate cme_w_x for each environment
        #print(type(train_data))
        #print(len(train_data))

        for env, d_data in enumerate(train_data):
            covars = {}
            covars['X'] = jnp.array(d_data['X'])

            cme_W_X = ConditionalMeanEmbed(jnp.array(d_data['W']), covars, self.lam_set['cme'], 
                                            kernel_dict=self.kernel_dict['cme_w_x'], scale=self.sc, 
                                            method=self.method_set['cme'])
            
            estimator['cme_w_x'][env]  = cme_W_X                     
            
        
        


        return estimator


    def _fit_target_domain(self, domain_data):
        #first fit k0 and cme_w_xz from the target domain

        #fit the conditional mean embedding for each domain
        estimator = {}

        covars = {}
        covars['X'] = jnp.array(domain_data['X'])

        cme_W_X = ConditionalMeanEmbed(jnp.array(domain_data['W']), covars, self.lam_set['cme'], 
                                        kernel_dict=self.kernel_dict['cme_w_x'], scale=self.sc, 
                                        method=self.method_set['cme'])
            
        estimator['cme_w_x']  = cme_W_X    
        return estimator


        
    def evaluation(self):
        eval_list = []
        n_env = len(self.source_test)
        
        target_testX = {}
        target_testX['X'] = self.target_test[0]['X']
        target_testY = self.target_test[0]['Y']
      
        for i in range(n_env):
            source_testX = {}
            source_testX['X'] = self.source_test[i]['X']
            source_testY = self.source_test[i]['Y']

            #source on source error
            predictY = self.predict(source_testX, 'source', 'source', i)
            ss_error = self.score(predictY, source_testY)
            eval_list.append({'task': 'source-source', 'env_id': i, 'predict error': ss_error})
        

            # source on target error
            predictY = self.predict(target_testX, 'source', 'source', i)
            st_error = self.score(predictY,  target_testY)
            eval_list.append({'task': 'source-target', 'env_id': i, 'predict error': st_error})
 
        # target on target errror
        predictY = self.predict(target_testX, 'target', 'target', 0)
        tt_error = self.score(predictY, target_testY)
        eval_list.append({'task': 'target-target', 'env_id': 0+n_env, 'predict error': tt_error})


        #adaptation error
        predictY = self.predict(target_testX, 'source', 'target', 0)
        adapt_error = self.score(predictY,  target_testY)
        eval_list.append({'task': 'adaptation', 'env_id': 0+n_env, 'predict error': adapt_error})

        df = pd.DataFrame(eval_list)
        print(df)

        return df



        #adaptation
    def predict(self, testX, k_domain, cme_domain, env_idx=0):
        if k_domain == 'source':
            k0 =  self.source_estimator['k0']
        else:
            k0 = self.target_estimator['k0']

        if cme_domain == 'source':
            cme_w_x = self.source_estimator['cme_w_x'][env_idx]
        else:
            cme_w_x = self.target_estimator['cme_w_x'][0]
        
        predictY = k0.get_EYx(testX, cme_w_x)
        return predictY


