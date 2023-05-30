import pandas as pd
import numpy as np

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta
from biogeme.models import loglogit, logit

from sklearn.model_selection import train_test_split
import lightgbm as lgb
from rumbooster import rum_train, RUMBooster

class swissmetro():

    def __init__(self, model_file = None):
        '''
        Class for the model related to swissmetro

        ----------
        parameters

        model_file: str
            file path to load a gbru model already saved
        '''
        self.dataset_path = 'Data/swissmetro.dat'
        self.dataset_name = 'SwissMetro'
        self.test_size = 0.2
        self.random_state = 42

        self.params = {'max_depth': 1, 
                       'num_boost_round': 300, 
                       'objective':'multiclass',
                       'monotone_constraints': [-1, -1, -1, -1, -1, -1, -1, -1], 
                       'interaction_constraints': [[0], [1], [2], [3], [4], [5], [6], [7]],
                       'learning_rate': 0.2,
                       'verbosity': 1,
                       'num_classes': 3
                      }

        self._load_preprocess_data()
        self._model()
        self._estimate_model()
        if model_file is not None:
            try: 
                self.gbru_model = RUMBooster(model_file=model_file)
                self.gbru_model.rum_structure = self._bio_to_rumboost()
                self.gbru_cross_entropy = self.gbru_model.best_score
            except:
                self._bio_rum_train()
        else:
            self._bio_rum_train()
        self._bio_predict()
        self._rum_predict()


    def _load_preprocess_data(self):
        '''
        Load and preprocess data
        '''
        df = pd.read_csv(self.dataset_path, sep='\t')
        keep = ((df['PURPOSE']!=1)*(df['PURPOSE']!=3)+(df['CHOICE']==0)) == 0
        df = df[keep]
        df.loc[:, 'TRAIN_COST'] = df['TRAIN_CO'] * (df['GA']==0)
        df.loc[:, 'SM_COST'] = df['SM_CO'] * (df['GA']==0)
        df_final = df[['TRAIN_TT', 'TRAIN_COST', 'TRAIN_HE', 'SM_TT', 'SM_COST', 'SM_HE', 'CAR_TT', 'CAR_CO', 'CHOICE']]
        df_train, df_test  = train_test_split(df_final, test_size=self.test_size, random_state=self.random_state)
        self.dataset_train = df_train
        self.dataset_test = df_test

    def _model(self):
        '''
        Create a MNL on the swissmetro dataset
        '''
        database_train = db.Database('swissmetro_train', self.dataset_train)

        globals().update(database_train.variables)

        # Parameters to be estimated
        ASC_CAR   = Beta('ASC_CAR', 0, None, None, 0)
        ASC_SM    = Beta('ASC_SM',  0, None, None, 0)
        ASC_TRAIN = Beta('ASC_SBB', 0, None, None, 1)

        B_TIME = Beta('B_TIME', 0, None, 0, 0)
        B_COST = Beta('B_COST', 0, None, 0, 0)
        B_HE   = Beta('B_HE',   0, None, 0, 0)

        # Utilities
        V_TRAIN = ASC_TRAIN + B_TIME * TRAIN_TT + B_COST * TRAIN_COST + B_HE * TRAIN_HE
        V_SM    = ASC_SM    + B_TIME * SM_TT    + B_COST * SM_COST    + B_HE * SM_HE
        V_CAR   = ASC_CAR   + B_TIME * CAR_TT   + B_COST * CAR_CO

        V = {1: V_TRAIN, 2: V_SM, 3: V_CAR}
        av = {1: 1, 2: 1, 3: 1}

        # Choice model estimation
        logprob = loglogit(V, av, CHOICE)
        biogeme = bio.BIOGEME(database_train, logprob)
        biogeme.modelName = "SwissmetroMNL"

        biogeme.generate_html = False
        biogeme.generate_pickle = False

        self.model = biogeme

    def _estimate_model(self):
        '''
        estimate a biogeme model from the biogeme.biogeme object
        '''
        #estimate model
        results = self.model.estimate()

        #results
        pandasResults = results.getEstimatedParameters()
        print(pandasResults)
        print(f"Nbr of observations: {self.model.database.getNumberOfObservations()}")
        print(f"LL(0) = {results.data.initLogLike:.3f}")
        print(f"LL(beta) = {results.data.logLike:.3f}")
        print(f"rho bar square = {results.data.rhoBarSquare:.3g}")
        print(f"Output file: {results.data.htmlFileName}")

        #cross entropy
        self.bio_cross_entropy = -results.data.logLike / self.model.database.getNumberOfObservations()
        self.betas = results.getBetaValues()

    

    def _bio_predict(self):
        '''
        predictions on the test set from the biogeme model
        '''
        database_test = db.Database('swissmetro_test', self.dataset_test)
        
        globals().update(database_test.variables)

        prob_train = logit(self.model.loglike.util, self.model.loglike.av, 1)
        prob_SM = logit(self.model.loglike.util, self.model.loglike.av, 2)
        prob_car = logit(self.model.loglike.util, self.model.loglike.av, 3)

        simulate ={'Prob. train': prob_train,
                   'Prob. SM':  prob_SM,
                   'Prob. car': prob_car}
        
        biogeme = bio.BIOGEME(database_test, simulate)
        biogeme.modelName = "swissmetro_logit_test"

        betaValues = self.betas

        self.bio_prediction = biogeme.simulate(betaValues)

        target = self.model.loglike.choice.name

        bioce_test = 0
        for i,l in enumerate(self.dataset_test[target]-1):
            bioce_test += np.log(self.bio_prediction.iloc[i,l])
        self.bio_cross_entropy_test = -bioce_test/len(self.dataset_test[target])

    def _rum_predict(self):
        '''
        predictions on the test set from the GBRU model
        '''
        target = self.model.loglike.choice.name
        features = [f for f in self.dataset_test.columns if f != target]
        test_data = lgb.Dataset(self.dataset_test.loc[:, features], label=self.dataset_test[[target]]-1, free_raw_data=False)
        self.gbru_prediction = self.gbru_model.predict(test_data)
        test_data.construct()
        self.gbru_cross_entropy_test = self.gbru_model.cross_entropy(self.gbru_prediction,test_data.get_label().astype(int))

    #TO IMPLEMENT
    #def hyperparameter_search()
    #add prediction on the test set for both models

    def _compare_models(self, on_test_set = True):
        '''
        compare one or several models estimated through biogeme and trained through GBRU, by calculating
        the cross-entropy on the train set.
        '''

        # print('On {}, biogeme has a negative CE of {} and GBRU of {} on the training set'.format(self.dataset_name, 
        #                                                                                          self.bio_cross_entropy,
        #                                                                                          self.gbru_cross_entropy))

        if on_test_set:
            
            self._bio_predict()
            self._rum_predict()
            print('On {}, biogeme has a negative CE of {} and RUMBooster of {} on the test set'.\
                  format(self.dataset_name,self.bio_cross_entropy_test,self.gbru_cross_entropy_test))
            

def process_parent(parent, pairs):
    if parent.getClassName() == 'Times':
        pairs.append(get_pair(parent))
    else:
        try:
            left = parent.left
            right = parent.right
        except:
            return pairs
        else:
            process_parent(left, pairs)
            process_parent(right, pairs)
        return pairs

def get_pair(parent):
    left = parent.left
    right = parent.right
    beta = None
    variable = None
    for exp in [left, right]:
        if exp.getClassName() == 'Beta':
            beta = exp.name
        elif exp.getClassName() == 'Variable':
            variable = exp.name
    if beta and variable:
        return (beta, variable)
    else:
        raise ValueError("Parent does not contain beta and variable")

def bio_to_rumboost(model):
    '''
    Converts a biogeme model to a rumboost dict
    '''
    utils = model.loglike.util
    rum_structure = []

    for k, v in utils.items():
        rum_structure.append({'columns': [], 'monotone_constraints': [], 'interaction_constraints': [], 'betas': [], 'categorical_feature': []})
        for i, pair in enumerate(process_parent(v, [])):
            rum_structure[-1]['columns'].append(pair[1])
            rum_structure[-1]['betas'].append(pair[0])
            rum_structure[-1]['interaction_constraints'].append([i])
            bounds = model.getBoundsOnBeta(pair[0])
            if (bounds[0] is None) and (bounds[1] is None):
                raise ValueError("Only one bound can be not None")
            if bounds[0] is not None:
                if bounds[0] >= 0:
                    rum_structure[-1]['monotone_constraints'].append(1)
            elif bounds[1] is not None:
                if bounds[1] <= 0:
                    rum_structure[-1]['monotone_constraints'].append(-1)
            else:
                rum_structure[k]['monotone_constraints'].append(0)
    return rum_structure

def rumb_train(model, params):
    rum_structure = bio_to_rumboost(model)
    data = model.database.data
    target = model.loglike.choice.name
    train_data = lgb.Dataset(data, label=data[target]-1, free_raw_data=False)
    model_trained = rum_train(params, train_data, valid_sets=[train_data], rum_structure=rum_structure)
    print('Maximum Likelihood value: {}'.format(model_trained.best_score*len(data[target])))
    return model_trained

def bio_predict(model, dataset_test, betas):
    '''
    predictions on the test set from the biogeme model
    '''
    database_test = db.Database('swissmetro_test', dataset_test)
    
    globals().update(database_test.variables)

    prob_train = logit(model.loglike.util, model.loglike.av, 1)
    prob_SM = logit(model.loglike.util, model.loglike.av, 2)
    prob_car = logit(model.loglike.util, model.loglike.av, 3)

    simulate ={'Prob. train': prob_train,
                'Prob. SM':  prob_SM,
                'Prob. car': prob_car}
    
    biogeme = bio.BIOGEME(database_test, simulate)
    biogeme.modelName = "swissmetro_logit_test"

    betaValues = betas

    bio_prediction = biogeme.simulate(betaValues)

    target = model.loglike.choice.name

    bioce_test = 0
    for i,l in enumerate(dataset_test[target]-1):
        bioce_test += np.log(bio_prediction.iloc[i,l])
    return -bioce_test/len(dataset_test[target])

def rum_predict(model, rumb_model, dataset_test):
    '''
    predictions on the test set from the GBRU model
    '''
    target = model.loglike.choice.name
    features = [f for f in dataset_test.columns if f != target]
    test_data = lgb.Dataset(dataset_test.loc[:, features], label=dataset_test[[target]]-1, free_raw_data=False)
    gbru_prediction = rumb_model.predict(test_data)
    test_data.construct()
    gbru_cross_entropy_test = rumb_model.cross_entropy(gbru_prediction,test_data.get_label().astype(int))
    return gbru_cross_entropy_test

def compare_models(model, rumb_model, dataset_test, betas):
    '''
    compare one or several models estimated through biogeme and trained through GBRU, by calculating
    the cross-entropy on the train set.
    '''

    bio_cross_entropy_test = bio_predict(model, dataset_test, betas)
    gbru_cross_entropy_test = rum_predict(model, rumb_model, dataset_test)
    print('On SwissMetro, biogeme has a negative CE of {} and RUMBooster of {} on the test set'.\
            format(bio_cross_entropy_test,gbru_cross_entropy_test))


