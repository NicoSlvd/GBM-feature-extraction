import pandas as pd
import numpy as np
import json
import pickle

import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.logging as blog
from biogeme.expressions import Beta
from biogeme.models import loglogit, logit

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

import lightgbm as lgb
from rumbooster import rum_train, RUMBooster, rum_cv
from utils import stratified_group_k_fold

class nts():

    def __init__(self, model_file = None):
        '''
        Class for the model related to NTS

        ----------
        parameters

        model_file: str
            file path to load a gbru model previously saved
        '''
        self.dataset_path_train = 'Data/nts_data_train.csv'
        self.dataset_path_test = 'Data/nts_data_test.csv'
        self.dataset_name = 'NTS'
        self.test_size = 0.2
        self.random_state = 42

        self.params = {'max_depth': 1, 
                       'num_boost_round': 1500, 
                       'learning_rate': 0.1,
                       'verbosity': 2,
                       'objective':'multiclass',
                       'num_classes': 4,
                       'early_stopping_round':50
                      }

        self._load_preprocess_data()
        self._model()

        if model_file is not None:
            self.gbru_model = RUMBooster(model_file=model_file)
            self.gbru_model.rum_structure = self._bio_to_rumboost()
            self.gbru_cross_entropy = self.gbru_model.best_score
            self._rum_predict()

    def _load_preprocess_data(self):
        '''
        Load and preprocess data
        '''
        #source: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice
        data_train = pd.read_csv(self.dataset_path_train)
        data_test = pd.read_csv(self.dataset_path_test)

        label_name = {'mode_main': 'choice'}
        self.dataset_train = data_train.rename(columns = label_name)
        self.dataset_test = data_test.rename(columns = label_name)

        target = 'choice'
        features = [f for f in self.dataset_test.columns if f != target]

        ind_id = np.array(data_train['individual_id'].values)

        train_idx = []
        test_idx = []
        try:
            train_idx, test_idx = pickle.load(open('nts_strat_group_k_fold.pickle', "rb"))
        except:
            for (train_i, test_i) in stratified_group_k_fold(data_train[features], data_train['mode_main'], ind_id, k=5):
                train_idx.append(train_i)
                test_idx.append(test_i)
            pickle.dump([train_idx, test_idx], open('nts_strat_group_k_fold.pickle', "wb"))

        self.folds = zip(train_idx, test_idx)

    def _model(self):
        '''
        Create a MNL on the NTS dataset
        Source of the model: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice
        '''
        database_train = db.Database('LTDS_train', self.dataset_train)

        logger = blog.get_screen_logger(level=blog.DEBUG)
        logger.info('Model NTS')

        globals().update(database_train.variables)
        MNL_beta_params_positive = ['B_cars_Car', 'B_bicycles_Bike', 'B_temp_Walk', 'B_temp_Bike']
        MNL_beta_params_negative = ['B_cars_Walk', 'B_cars_Bike', 'B_cars_Public_Transport', 'B_bicycles_Walk', 'B_bicycles_Public_Transport', 'B_bicycles_Car', 'B_precip_Walk', 'B_precip_Bike']
        MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_density_Walk', 'B_density_Bike', 'B_density_Public_Transport', 'B_density_Car', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car','B_diversity_Walk', 'B_diversity_Bike', 'B_diversity_Public_Transport', 'B_diversity_Car', 'B_green_Walk', 'B_green_Bike', 'B_green_Public_Transport', 'B_green_Car', 'B_temp_Public_Transport', 'B_temp_Car', 'B_precip_Public_Transport', 'B_precip_Car', 'B_wind_Walk', 'B_wind_Bike', 'B_wind_Public_Transport', 'B_wind_Car', 'B_license_cat_Walk', 'B_license_cat_Bike', 'B_license_cat_Public_Transport', 'B_license_cat_Car', 'B_male_cat_Walk', 'B_male_cat_Bike', 'B_male_cat_Public_Transport', 'B_male_cat_Car', 'B_weekend_cat_Walk', 'B_weekend_cat_Bike', 'B_weekend_cat_Public_Transport', 'B_weekend_cat_Car', 'B_native_Walk', 'B_native_Bike', 'B_native_Public_Transport', 'B_native_Car', 'B_nonwestern_Walk', 'B_nonwestern_Bike', 'B_nonwestern_Public_Transport', 'B_nonwestern_Car', 'B_western_Walk', 'B_western_Bike', 'B_western_Public_Transport', 'B_western_Car', 'B_income_cat_Walk', 'B_income_cat_Bike', 'B_income_cat_Public_Transport', 'B_income_cat_Car', 'B_education_cat_Walk', 'B_education_cat_Bike', 'B_education_cat_Public_Transport', 'B_education_cat_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car']
        MNL_utilities = {0: 'B_distance_Walk*distance + B_density_Walk*density + B_age_Walk*age + B_cars_Walk*cars + B_bicycles_Walk*bicycles + B_diversity_Walk*diversity + B_green_Walk*green + B_temp_Walk*temp + B_precip_Walk*precip + B_wind_Walk*wind + B_license_cat_Walk*license_cat + B_male_cat_Walk*male_cat + B_weekend_cat_Walk*weekend_cat + B_native_Walk*native + B_nonwestern_Walk*nonwestern + B_western_Walk*western + B_income_cat_Walk*income_cat + B_education_cat_Walk*education_cat',
                         1: 'ASC_Bike + B_distance_Bike*distance + B_density_Bike*density + B_age_Bike*age + B_cars_Bike*cars + B_bicycles_Bike*bicycles + B_diversity_Bike*diversity + B_green_Bike*green + B_temp_Bike*temp + B_precip_Bike*precip + B_wind_Bike*wind + B_license_cat_Bike*license_cat + B_male_cat_Bike*male_cat + B_weekend_cat_Bike*weekend_cat + B_native_Bike*native + B_nonwestern_Bike*nonwestern + B_western_Bike*western + B_income_cat_Bike*income_cat + B_education_cat_Bike*education_cat',
                         2: 'ASC_Public_Transport + B_distance_Public_Transport*distance + B_density_Public_Transport*density + B_age_Public_Transport*age + B_cars_Public_Transport*cars + B_bicycles_Public_Transport*bicycles + B_diversity_Public_Transport*diversity + B_green_Public_Transport*green + B_temp_Public_Transport*temp + B_precip_Public_Transport*precip + B_wind_Public_Transport*wind + B_license_cat_Public_Transport*license_cat + B_male_cat_Public_Transport*male_cat + B_weekend_cat_Public_Transport*weekend_cat + B_native_Public_Transport*native + B_nonwestern_Public_Transport*nonwestern + B_western_Public_Transport*western + B_income_cat_Public_Transport*income_cat + B_education_cat_Public_Transport*education_cat',
                         3: 'ASC_Car + B_distance_Car*distance + B_density_Car*density + B_age_Car*age + B_cars_Car*cars + B_bicycles_Car*bicycles + B_diversity_Car*diversity + B_green_Car*green + B_temp_Car*temp + B_precip_Car*precip + B_wind_Car*wind + B_license_cat_Car*license_cat + B_male_cat_Car*male_cat + B_weekend_cat_Car*weekend_cat + B_native_Car*native + B_nonwestern_Car*nonwestern + B_western_Car*western + B_income_cat_Car*income_cat + B_education_cat_Car*education_cat'}
        # Construct the model parameters
        for beta in MNL_beta_params_positive:
            exec("{} = Beta('{}', 0, 0, None, 0)".format(beta, beta), globals())
        for beta in MNL_beta_params_negative:
            exec("{} = Beta('{}', 0, None, 0, 0)".format(beta, beta), globals())
        for beta in MNL_beta_params_neutral:
            exec("{} = Beta('{}', 0, None, None, 0)".format(beta, beta), globals())

        # Define utility functions
        for utility_idx in MNL_utilities.keys():
            exec("V_{} = {}".format(utility_idx, MNL_utilities[utility_idx]), globals())

        # Assign utility functions to utility indices
        exec("V_dict = {}", globals())
        for utility_idx in MNL_utilities.keys():
            exec("V_dict[{}] = V_{}".format(utility_idx, utility_idx), globals())

        # Associate the availability conditions with the alternatives
        exec("av = {}", globals())
        for utility_idx in MNL_utilities.keys():
            exec("av[{}] = 1".format(utility_idx), globals())
        
        # Definition of the model. This is the contribution of each
        # observation to the log likelihood function.
        logprob = loglogit(V_dict, av, choice)

        # Create the Biogeme object
        biogeme = bio.BIOGEME(database_train, logprob)
        biogeme.modelName = 'Biogeme-model'

        biogeme.generate_html = False
        biogeme.generate_pickle = False

        self.model = biogeme

    def estimate_model(self):
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

    def _process_parent(self, parent, pairs):
        '''
        Dig into the raw biogeme utility expressions until arriving to a beta times variable expression
        '''

        if parent.getClassName() == 'Times':
            pairs.append(self._get_pair(parent))
        else:
            try:
                left = parent.left
                right = parent.right
            except:
                return pairs
            else:
                self._process_parent(left, pairs)
                self._process_parent(right, pairs)
            return pairs
    
    def _get_pair(self, parent):
        '''
        Get beta and the variable associated
        '''
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
        
    def _bio_to_rumboost(self):
        '''
        Converts a biogeme model to a rumboost dict
        '''
        utils = self.model.loglike.util
        rum_structure = []
        
        for k, v in utils.items():
            rum_structure.append({'columns': [], 'monotone_constraints': [], 'interaction_constraints': [], 'betas': []})
            for i, pair in enumerate(self._process_parent(v, [])):
                rum_structure[-1]['columns'].append(pair[1])
                rum_structure[-1]['betas'].append(pair[0])
                rum_structure[-1]['interaction_constraints'].append([i])
                bounds = self.model.getBoundsOnBeta(pair[0])
                if (bounds[0] is not None) and (bounds[1] is not None):
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

    def bio_rum_train(self, valid_test = False):
        rum_structure = self._bio_to_rumboost()
        self.params['learning_rate'] = 0.1
        self.params['early_stopping_rounds'] = 50
        self.params['num_boost_round'] = 1500
        self.params['lambda_l2'] = 0
        self.params['boosting'] = 'gbdt'
        self.params['feature_fraction'] = 1
        self.params['monotone_constraints_method'] =  'advanced'
        data = self.model.database.data
        target = self.model.loglike.choice.name
        train_data = lgb.Dataset(data, label=data[target], free_raw_data=False)
        validate_data = lgb.Dataset(self.dataset_test, label=self.dataset_test[target], free_raw_data=False)
        if not valid_test:
            model_rumtrained = rum_train(self.params, train_data, valid_sets=[train_data], rum_structure=rum_structure)
        else:
            model_rumtrained = rum_train(self.params, train_data, valid_sets=[validate_data], rum_structure=rum_structure)
        self.gbru_cross_entropy = model_rumtrained.best_score
        self.gbru_model = model_rumtrained

        self.gbru_model.save_model('nts_gbru_model.json')

    def hyperparameter_optim(self):
        '''
        hyperparameter fine tuning
        '''
        learning_rates = [0.15]
        self.params['early_stopping_rounds'] = 20
        self.params['num_boost_round'] = 1500
        self.params['lambda_l2'] = 0.1
        self.params['boosting'] = 'goss'
        self.params['feature_fraction'] = 0.8
        self.params['monotone_constraints_method'] =  'advanced'
        rum_structure = self._bio_to_rumboost()
        data = self.model.database.data
        target = self.model.loglike.choice.name
        train_data = lgb.Dataset(data, label=data[target], free_raw_data=False)
        best_score = 1000

        for l in learning_rates:
            self.params['learning_rate'] = l
            cv_gbru = rum_cv(self.params, train_data, folds=self.folds, verbose_eval=True, rum_structure=rum_structure)
            bs = np.min(cv_gbru['Cross entropy --- mean'])
            bi = len(cv_gbru['Cross entropy --- mean'])
            if bs < best_score:
                best_score = bs
                best_lr = l
                best_iter = bi
        self.params['learning_rate'] = best_lr
        self.gbru_cross_entropy = best_score
        self.params['num_boost_round'] = bi

        with open('best_lr.txt', 'w') as f:
            json.dump(best_lr, f)
        with open('best_score.txt', 'w') as f:
            json.dump(best_score, f)
        with open('best_iter.txt', 'w') as f:
            json.dump(best_iter, f)

    def _bio_predict(self):
        '''
        predictions on the test set from the biogeme model
        '''
        database_test = db.Database('LTDS_test', self.dataset_test)
        
        globals().update(database_test.variables)

        prob_walk = logit(self.model.loglike.util, self.model.loglike.av, 0)
        prob_bike = logit(self.model.loglike.util, self.model.loglike.av, 1)
        prob_pt = logit(self.model.loglike.util, self.model.loglike.av, 2)
        prob_car = logit(self.model.loglike.util, self.model.loglike.av, 3)

        simulate ={'Prob. walk': prob_walk,
                   'Prob. bike': prob_bike,
                   'Prob. PT':   prob_pt,
                   'Prob. car':  prob_car}
        
        biogeme = bio.BIOGEME(database_test, simulate)
        biogeme.modelName = "nts_logit_test"

        betaValues = self.betas

        self.bio_prediction = biogeme.simulate(betaValues)

        target = self.model.loglike.choice.name

        bioce_test = 0
        for i,l in enumerate(self.dataset_test[target]):
            bioce_test += np.log(self.bio_prediction.iloc[i,l])
        self.bio_cross_entropy_test = -bioce_test/len(self.dataset_test[target])

    def _rum_predict(self):
        '''
        predictions on the test set from the GBRU model
        '''
        target = self.model.loglike.choice.name
        features = [f for f in self.dataset_test.columns if f != target]
        test_data = lgb.Dataset(self.dataset_test.loc[:, features], label=self.dataset_test[[target]], free_raw_data=False)
        self.gbru_prediction = self.gbru_model.predict(test_data)
        test_data.construct()
        self.gbru_cross_entropy_test = self.gbru_model.cross_entropy(self.gbru_prediction,test_data.get_label().astype(int))
        self.gbru_accuracy_test = self.gbru_model.accuracy(self.gbru_prediction,test_data.get_label().astype(int))

    #TO IMPLEMENT
    #def hyperparameter_search()
    #add prediction on the test set for both models

    def compare_models(self, on_test_set = False):
        '''
        compare one or several models estimated through biogeme and trained through GBRU, by calculating
        the cross-entropy on the train set.
        '''
        #print('On {}, biogeme has a negative CE of {} and GBRU of {} on the training set'.format(self.dataset_name, 
        #                                                                                         self.bio_cross_entropy,
        #                                                                                         self.gbru_cross_entropy))

        #if on_test_set:
        #    print('On {}, biogeme has a negative CE of {} and GBRU of {} on the test set'.\
        #          format(self.dataset_name,self.bio_cross_entropy_test,self.gbru_cross_entropy_test))
            
        print('On {}, biogeme has a negative CE of .. and GBRU of {} on the training set'.format(self.dataset_name, 
                                                                                                 self.gbru_cross_entropy))

        if on_test_set:
            self._rum_predict()
            print('On {}, biogeme has a negative CE of .. and GBRU of {} on the test set'.\
                  format(self.dataset_name,self.gbru_cross_entropy_test))

