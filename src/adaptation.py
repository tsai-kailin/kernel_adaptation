#this file implement the full adaptation pipeline
import pandas as pd
import numpy as np

from method import KernelMethod
from cme import ConditionalMeanEmbed
from bridge_h0 import Bridge_h0
from bridge_m0 import CME_m0_cme

class full_adapt(KernelMethod):
    """
    Adaptation setting: observe (W,X,Y,C) from the source, (W,X,C) from the target

    """
    def split_data(self):
        #split training data
        n = len(self.source_train)

        train1, train2, train3 = np.split(self.source_train.sample(frac=1, random_state=42), [int(.33*n), int(.67*n)])
        self.soure_train = [pd.DataFrame(train1), pd.DataFrame(train2), pd.DataFrame(train3)]

        n2 = len(self.target_train)
        train1, train2, train3 = np.split(self.target_train.sample(frac=1, random_state=42), [int(.33*n2), int(.67*n2)])
        self.target_train = [pd.DataFrame(train1), pd.DataFrame(train2), pd.DataFrame(train3)]


    def _fit_one_domain(self, domain_data):
        """ fit single domain.
        Args:
            domain_data: data to train, pandas.DataFrame or list of pandas.DataFrame
        """
        if self.split:
            train_data = self.domain_data[0]
        else:
            train_data = self.domain_data
        
        covars = {}
        covars['X'] = train_data['X']
        covars['C'] = train_data['C']

        cme_W_XC = ConditionalMeanEmbed(train_data['W'], covars, self.lam_set['cme'], self.sc, method=self.method_set['cme'])

        # estimate cme(W,C|x)
        if self.split:
            train_data = self.domain_data[1]
        else:
            train_data = self.domain_data
        covars = {}
        covars['X'] = train_data['X']
        WC = jnp.hstack((train_data['W'], train_data['C']))
        cme_WC_X = ConditionalMeanEmbed(WC, covars, self.lam['cme'], self.sc,  method=self.method_set['cme'])



        # estimate h0
        Xlist = cme_W_XC.get_params()['Xlist']
        if self.split:
            train_data = self.domain_data[2]
        else:
            train_data = self.domain_data 
        
        covars = {}
        for key in Xlist:
            covars[key] = train_data[key]
        h0 = Bridge_h0(cme_W_XC, covars, train_data['Y'], self.lam['h0'], self.sc,  method=self.method_set['h0'])
        estimator = {}
        estimate['cme_w_xc'] = cme_W_XC
        estimate['cme_wc_x'] = cme_WC_X
        estimate['h0'] = h0

        return estimator


        
    def evaluation(self):
        eval_list = []

        #source evaluation
        source_testX = self.source_test['X']
        source_testY = self.source_test['Y']

        #source on source error
        predictY = self.predict(source_textX, 'source', 'source')
        ss_error = self.score(predictY, source_testY)
        eval_list.append({'task': 'source-source', 'predict error': ss_error})
        


        # target on source error
        predictY = self.predict(source_textX, 'target', 'target')
        ts_error = self.score(predictY, source_testY)
        eval_list.append({'task': 'target-source', 'predict error': ts_error})

        #target evaluation
        target_testX = self.target_test['X']
        target_testY = self.target_test['Y']
      
        # target on target errror
        predictY = self.predict(target_textX, 'target', 'target')
        tt_error = self.score(predictY, target_testY)
        eval_list.append({'task': 'target-target', 'predict error': tt_error})

        # source on target error
        predictY = self.predict(target_testX, 'source', 'source')
        st_error = self.score(predictY,  target_testY)
        eval_list.append({'task': 'source-target', 'predict error': st_error})


        #adaptation error
        predictY = self.predict(target_testX, 'source', 'target')
        adapt_error = self.score(predictY,  target_testY)
        eval_list.append({'task': 'adaptation', 'predict error': adapt_error})

        df = pd.DataFrame(eval_list)
        print(df)



        #adaptation
    def predict(self, testX, h_domain, cme_domain):
        if h_domain == 'source':
            h0 =  self.source_estimator['h0']
        else:
            h0 = self.target_estimator['h0']

        if cme_domain == 'source':
            cme_wc_x = self.source_estimator['cme_wc_x']
        else:
            cme_wc_x = self.target_estimator['cme_wc_x']
        
        predictY = h0.get_EYx(testX, cme_wc_x)
        return predictY




class partial_adapt(KernelMethod):
    """
    Adaptation setting: observe (W,X,Y,C) from the source, (W,X,C) from the target
    """
    def split_data(self):
        #split to four batches evenly
        n = len(self.source_train)

        train1, train2, train3, train4 = np.split(self.source_train.sample(frac=1, random_state=42), [int(.25*n), int(.5*n), int(.75*n)])
        self.soure_train = [pd.DataFrame(train1), pd.DataFrame(train2), pd.DataFrame(train3), pd.DataFrame(train3)]

        n2 = len(self.target_train)
        train1, train2, train3, train4 = np.split(self.target_train.sample(frac=1, random_state=42), [int(.25*n), int(.5*n), int(.75*n)])
        self.target_train = [pd.DataFrame(train1), pd.DataFrame(train2), pd.DataFrame(train3), pd.DataFrame(train4)]


    def _fit_one_domain(self, domain_data):
        """ fit single domain.
        Args:
            domain_data: data to train, pandas.DataFrame or list of pandas.DataFrame
        """
        if self.split:
            train_data = self.domain_data[0]
        else:
            train_data = self.domain_data
        
        covars = {}
        covars['X'] = train_data['X']
        covars['C'] = train_data['C']

        cme_W_XC = ConditionalMeanEmbed(train_data['W'], covars, self.lam_set['cme'], self.sc, method=self.method_set['cme'])

        # estimate cme(W|x) and cme(C|x)
        if self.split:
            train_data = self.domain_data[1]
        else:
            train_data = self.domain_data
        covars = {}
        covars['X'] = train_data['X']

        cme_W_X = ConditionalMeanEmbed(train_data['W'], covars, self.lam['cme'], self.sc,  method=self.method_set['cme'])
        cme_C_X = ConditionalMeanEmbed(train_data['C'], covars, self.lam['cme'], self.sc,  method=self.method_set['cme'])


        # estimate m0
        if self.split:
            train_data = self.domain_data[3]
        else:
            train_data = self.domain_data

        covars = {}
        covars['X'] = train_data['X']
        covars['C'] = train_data['C']


        m0 =  CME_m0_cme(cme_W_X, covars, self.lam['m0'], self.sc,  method=self.method_set['m0'])


        # estimate h0
        Xlist = cme_W_XC.get_params()['Xlist']
        if self.split:
            train_data = self.domain_data[3]
        else:
            train_data = self.domain_data 
        
        covars = {}
        for key in Xlist:
            covars[key] = train_data[key]
        h0 = Bridge_h0(cme_W_XC, covars, train_data['Y'], self.lam['h0'], self.sc,  method=self.method_set['h0'])
        estimator = {}
        estimate['cme_w_xc'] = cme_W_XC
        estimate['cme_w_x'] = cme_W_X
        estimate['cme_c_x'] = cme_C_X
        estimate['h0'] = h0
        estimate['m0'] = m0

        return estimator

    def predict(self, testX, h_domain, cme_domain):
        if h_domain == 'source':
            h0 =  self.source_estimator['h0']
        else:
            h0 = self.target_estimator['h0']

        if cme_domain == 'source':
            cme_w_x = self.source_estimator['cme_w_x']
            cme_c_x = self.source_estimator['cme_c_x']
        else:
            cme_w_x = self.target_estimator['cme_w_x']
            cme_c_x = self.target_estimator['cme_c_x']
        
        predictY = h0.get_EYx_independent_cme(testX, cme_w_x, cme_c_x)
        return predictY

    def predict_adapt(self, testX, h_domain, cme_domain, m_domain)
        if h_domain == 'source':
            h0 =  self.source_estimator['h0']
        else:
            h0 = self.target_estimator['h0']

        if cme_domain == 'source':
            cme_w_x = self.source_estimator['cme_w_x']
        else:
            cme_w_x = self.target_estimator['cme_w_x']
        
        if m_domain == 'source':
            m0 = self.source_estimator['m0']
        else:
            m0 = self.target_estimator['m0']
        predictY = h0.get_EYx_independent(testX, cme_w_x, m0)

        return predictY
        
    def evaluation(self):
        eval_list = []

        #source evaluation
        source_testX = self.source_test['X']
        source_testY = self.source_test['Y']

        #source on source error
        predictY = self.predict(source_textX, 'source', 'source')
        ss_error = self.score(predictY, source_testY)
        eval_list.append({'task': 'source-source', 'predict error': ss_error})
        
        # target on source error
        predictY = self.predict(source_textX, 'target', 'target')
        ts_error = self.score(predictY, source_testY)
        eval_list.append({'task': 'target-source', 'predict error': ts_error})

        #target evaluation
        target_testX = self.target_test['X']
        target_testY = self.target_test['Y']
      
        # target on target errror
        predictY = self.predict(target_textX, 'target', 'target')
        tt_error = self.score(predictY, target_testY)
        eval_list.append({'task': 'target-target', 'predict error': tt_error})

        # source on target error
        predictY = self.predict(target_testX, 'source', 'source')
        st_error = self.score(predictY,  target_testY)
        eval_list.append({'task': 'source-target', 'predict error': st_error})


        #adaptation error
        predictY = self.predict_adapt(target_testX, 'source', 'target', 'source')
        adapt_error = self.score(predictY,  target_testY)
        eval_list.append({'task': 'adaptation', 'predict error': adapt_error})

        df = pd.DataFrame(eval_list)
        print(df)