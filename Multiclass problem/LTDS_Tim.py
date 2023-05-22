import pandas as pd
import numpy as np
import json
import pickle

import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.logging as blog
from biogeme.expressions import Beta
from biogeme.models import loglogit, logit

import lightgbm as lgb
from rumbooster import rum_train, RUMBooster, rum_cv
from utils import stratified_group_k_fold

class ltds_54():

    def __init__(self, model_file = None):
        '''
        Class for the model related to LTDS

        ----------
        parameters

        model_file: str
            file path to load a gbru model already saved
        '''
        self.dataset_path_train = 'Data/LTDS_train.csv'
        self.dataset_path_test = 'Data/LTDS_test.csv'
        self.dataset_name = 'LTDS'

        self.params = {'max_depth': 1, 
                       'num_boost_round': 1500, 
                       'learning_rate': 0.1,
                       'verbosity': 1,
                       'objective':'multiclass',
                       'num_classes': 4,
                       'early_stopping_round':50,
                      }

        self._load_preprocess_data()
        self._model()

        if model_file is not None:
            self.gbru_model = RUMBooster(model_file=model_file)
            self.gbru_model.rum_structure = self._bio_to_rumboost()
            self.gbru_cross_entropy = self.gbru_model.best_score
            bio_database_test = self._bio_database(self.dataset_test, 'LTDS_test')
            self._new_variables(bio_database_test)
            self._rum_predict(data_test = bio_database_test.data)


    def _load_preprocess_data(self):
        '''
        Load and preprocess data
        '''
        #source: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice
        data_train = pd.read_csv(self.dataset_path_train)
        data_test = pd.read_csv(self.dataset_path_test)

        label_name = {'travel_mode': 'choice'}
        self.dataset_train = data_train.rename(columns = label_name)
        self.dataset_test = data_test.rename(columns = label_name)

        target = 'choice'
        features = [f for f in self.dataset_test.columns if f != target]

        hh_id = np.array(data_train['household_id'].values)

        train_idx = []
        test_idx = []
        try:
            train_idx, test_idx = pickle.load(open('strat_group_k_fold.pickle', "rb"))
        except FileNotFoundError:
            for (train_i, test_i) in stratified_group_k_fold(data_train[features], data_train['travel_mode'], hh_id, k=5):
                train_idx.append(train_i)
                test_idx.append(test_i)
            pickle.dump([train_idx, test_idx], open('strat_group_k_fold.pickle', "wb"))

        self.folds = zip(train_idx, test_idx)

    def _bio_database(self, data, database_name):
        '''
        Transform a pandas Dataframe into a biogeme database object
        '''
        database = db.Database(database_name, data)
        globals().update(database.variables)

        return database

    def _new_variables(self, database_train):
        '''
        Create the new variables for the model
        '''

        co1 = database_train.DefineVariable('co1', car_ownership == 1)
        co2 = database_train.DefineVariable('co2', car_ownership == 2)
        weekday = database_train.DefineVariable('weekday', day_of_week < 6)
        saturday = database_train.DefineVariable('saturday', day_of_week == 6)
        child = database_train.DefineVariable('child', age < 18)
        pensioner = database_train.DefineVariable('pensioner', age > 64)
        winter = database_train.DefineVariable('winter', travel_month < 3 or travel_month == 12)
        # ampeak = database_train.DefineVariable('ampeak', self.start_time_linear<9.5 and self.start_time_linear>=6.5, database_train)
        pmpeak = database_train.DefineVariable('pmpeak', start_time_linear < 19.5 and start_time_linear >= 16.5)
        interpeak = database_train.DefineVariable('interpeak', start_time_linear < 16.5 and start_time_linear >= 9.5)
        distance_km = database_train.DefineVariable('distance_km', distance / 1000)
        drive_cost = database_train.DefineVariable('drive_cost', cost_driving_total * (car_ownership > 0))

        globals().update(database_train.variables)


    def _model(self):
        '''
        Create a MNL on the LTDS dataset
        Source of the model: https://github.com/glederrey/HAMABS/blob/master/code/models/LPMC_RR.py
        '''
        
        database_train = self._bio_database(self.dataset_train, 'LTDS_train')

        logger = blog.get_screen_logger(level=blog.DEBUG)
        logger.info('Model LTDS.py')

        #ASC_WALKING = Beta('ASC_WALKING', 0, None, None, 1)
        ASC_CYCLING = Beta('ASC_CYCLING', 0, None, None, 0)
        ASC_PT = Beta('ASC_PT', 0, None, None, 0)
        ASC_DRIVING = Beta('ASC_DRIVING', 0, None, None, 0)

        B_TIME_WALKING = Beta('B_TIME_WALKING', 0, None, 0, 0)
        B_TIME_CYCLING = Beta('B_TIME_CYCLING', 0, None, 0, 0)
        B_TIME_DRIVING = Beta('B_TIME_DRIVING', 0, None, 0, 0)
        B_TRAFFIC_DRIVING = Beta('B_TRAFFIC_DRIVING', 0, None, 0, 0)

        B_COST_DRIVE = Beta('B_COST_DRIVE', 0, None, 0, 0)
        B_COST_PT = Beta('B_COST_PT', 0, None, 0, 0)

        B_TIME_PT_BUS = Beta('B_TIME_PT_BUS', 0, None, 0, 0)
        B_TIME_PT_RAIL = Beta('B_TIME_PT_RAIL', 0, None, 0, 0)
        B_TIME_PT_ACCESS = Beta('B_TIME_PT_ACCESS', 0, None, 0, 0)
        B_TIME_PT_INT_WALK = Beta('B_TIME_PT_INT_WALK', 0, None, 0, 0)
        B_TIME_PT_INT_WAIT = Beta('B_TIME_PT_INT_WAIT', 0, None, 0, 0)

        #B_PURPOSE_B_WALKING = Beta('B_PURPOSE_B_WALKING', 0, None, None, 1)
        B_PURPOSE_B_CYCLING = Beta('B_PURPOSE_B_CYCLING', 0, None, None, 0)
        B_PURPOSE_B_PT = Beta('B_PURPOSE_B_PT', 0, None, None, 0)
        B_PURPOSE_B_DRIVING = Beta('B_PURPOSE_B_DRIVING', 0, None, None, 0)
        #B_PURPOSE_HBW_WALKING = Beta('B_PURPOSE_HBW_WALKING', 0, None, None, 1)
        B_PURPOSE_HBW_CYCLING = Beta('B_PURPOSE_HBW_CYCLING', 0, None, None, 0)
        B_PURPOSE_HBW_PT = Beta('B_PURPOSE_HBW_PT', 0, None, None, 0)
        B_PURPOSE_HBW_DRIVING = Beta('B_PURPOSE_HBW_DRIVING', 0, None, None, 0)
        #B_PURPOSE_HBE_WALKING = Beta('B_PURPOSE_HBE_WALKING', 0, None, None, 1)
        # B_PURPOSE_HBE_CYCLING = Beta('B_PURPOSE_HBE_CYCLING',0,-10,10,0)
        B_PURPOSE_HBE_PT = Beta('B_PURPOSE_HBE_PT', 0, None, None, 0)
        B_PURPOSE_HBE_DRIVING = Beta('B_PURPOSE_HBE_DRIVING', 0, None, None, 0)
        #B_PURPOSE_HBO_WALKING = Beta('B_PURPOSE_HBO_WALKING', 0, None, None, 1)
        B_PURPOSE_HBO_CYCLING = Beta('B_PURPOSE_HBO_CYCLING', 0, None, None, 0)
        B_PURPOSE_HBO_PT = Beta('B_PURPOSE_HBO_PT', 0, None, None, 0)
        # B_PURPOSE_HBO_DRIVING = Beta('B_PURPOSE_HBO_DRIVING',0,-10,10,0)

        #B_VEHICLE_OWNERSHIP_1_WALKING = Beta('B_VEHICLE_OWNERSHIP_1_WALKING', 0, None, None, 1)
        B_VEHICLE_OWNERSHIP_CYCLING = Beta('B_VEHICLE_OWNERSHIP_CYCLING', 0, None, None, 0)
        B_VEHICLE_OWNERSHIP_1_PT = Beta('B_VEHICLE_OWNERSHIP_1_PT', 0, None, 0, 0)
        B_VEHICLE_OWNERSHIP_1_DRIVING = Beta('B_VEHICLE_OWNERSHIP_1_DRIVING', 0, 0, None, 0)
        #B_VEHICLE_OWNERSHIP_2_WALKING = Beta('B_VEHICLE_OWNERSHIP_2_WALKING', 0, None, None, 1)
        B_VEHICLE_OWNERSHIP_2_PT = Beta('B_VEHICLE_OWNERSHIP_2_PT', 0, None, 0, 0)
        B_VEHICLE_OWNERSHIP_2_DRIVING = Beta('B_VEHICLE_OWNERSHIP_2_DRIVING', 0, 0, None, 0)

        #B_DRIVING_LICENCE_WALKING = Beta('B_DRIVING_LICENCE_WALKING', 0, None, None, 1)
        B_DRIVING_LICENCE_CYCLING = Beta('B_DRIVING_LICENCE_CYCLING', 0, None, 0, 0)
        B_DRIVING_LICENCE_PT = Beta('B_DRIVING_LICENCE_PT', 0, None, 0, 0)
        B_DRIVING_LICENCE_DRIVING = Beta('B_DRIVING_LICENCE_DRIVING', 0, 0, None, 0)

        #B_FEMALE_WALKING = Beta('B_FEMALE_WALKING', 0, None, None, 1)
        B_FEMALE_CYCLING = Beta('B_FEMALE_CYCLING', 0, None, None, 0)
        B_FEMALE_PT = Beta('B_FEMALE_PT', 0, None, None, 0)
        B_FEMALE_DRIVING = Beta('B_FEMALE_DRIVING', 0, None, None, 0)

        #B_WINTER_WALKING = Beta('B_WINTER_WALKING', 0, None, None, 1)
        B_WINTER_CYCLING = Beta('B_WINTER_CYCLING', 0, None, 0, 0)
        # B_WINTER_PT = Beta('B_WINTER_PT',0,-10,10,0)
        B_WINTER_DRIVING = Beta('B_WINTER_DRIVING', 0, None, None, 0)

        #B_DISTANCE_WALKING = Beta('B_DISTANCE_WALKING', 0, None, None, 1)
        B_DISTANCE_CYCLING = Beta('B_DISTANCE_CYCLING', 0, None, None, 0)
        B_DISTANCE_PT = Beta('B_DISTANCE_PT', 0, None, None, 0)
        B_DISTANCE_DRIVING = Beta('B_DISTANCE_DRIVING', 0, None, None, 0)

        #B_AGE_CHILD_WALKING = Beta('B_AGE_CHILD_WALKING', 0, None, None, 1)
        # B_AGE_CHILD_CYCLING = Beta('B_AGE_CHILD_CYCLING',0,-10,10,0)
        B_AGE_CHILD_PT = Beta('B_AGE_CHILD_PT', 0, None, None, 0)
        B_AGE_CHILD_DRIVING = Beta('B_AGE_CHILD_DRIVING', 0, None, None, 0)
        #B_AGE_PENSIONER_WALKING = Beta('B_AGE_PENSIONER_WALKING', 0, None, None, 1)
        B_AGE_PENSIONER_CYCLING = Beta('B_AGE_PENSIONER_CYCLING', 0, None, None, 0)
        B_AGE_PENSIONER_PT = Beta('B_AGE_PENSIONER_PT', 0, None, None, 0)
        B_AGE_PENSIONER_DRIVING = Beta('B_AGE_PENSIONER_DRIVING', 0, None, None, 0)

        #B_DAY_WEEK_WALKING = Beta('B_DAY_WEEK_WALKING', 0, None, None, 1)
        # B_DAY_WEEK_CYCLING = Beta('B_DAY_WEEK_CYCLING',0,-10,10,0)
        B_DAY_WEEK_PT = Beta('B_DAY_WEEK_PT', 0, None, None, 0)
        B_DAY_WEEK_DRIVING = Beta('B_DAY_WEEK_DRIVING', 0, None, None, 0)
        #B_DAY_SAT_WALKING = Beta('B_DAY_SAT_WALKING', 0, None, None, 1)
        B_DAY_SAT_CYCLING = Beta('B_DAY_SAT_CYCLING', 0, None, None, 0)
        B_DAY_SAT_PT = Beta('B_DAY_SAT_PT', 0, None, None, 0)
        # B_DAY_SAT_DRIVING = Beta('B_DAY_SAT_DRIVING',0,-10,10,0)

        # B_DEPARTURE_AM_PEAK_WALKING = Beta('B_DEPARTURE_AM_PEAK_WALKING',0,-10,10,1)
        # B_DEPARTURE_AM_PEAK_CYCLING = Beta('B_DEPARTURE_AM_PEAK_CYCLING',0,-10,10,0)
        # B_DEPARTURE_AM_PEAK_PT = Beta('B_DEPARTURE_AM_PEAK_PT',0,-10,10,0)
        # B_DEPARTURE_AM_PEAK_DRIVING = Beta('B_DEPARTURE_AM_PEAK_DRIVING',0,-10,10,0)
        #B_DEPARTURE_PM_PEAK_WALKING = Beta('B_DEPARTURE_PM_PEAK_WALKING', 0, None, None, 1)
        B_DEPARTURE_PM_PEAK_CYCLING = Beta('B_DEPARTURE_PM_PEAK_CYCLING', 0, None, None, 0)
        B_DEPARTURE_PM_PEAK_PT = Beta('B_DEPARTURE_PM_PEAK_PT', 0, None, None, 0)
        B_DEPARTURE_PM_PEAK_DRIVING = Beta('B_DEPARTURE_PM_PEAK_DRIVING', 0, None, None, 0)
        #B_DEPARTURE_INTER_PEAK_WALKING = Beta('B_DEPARTURE_INTER_PEAK_WALKING', 0, None, None, 1)
        B_DEPARTURE_INTER_PEAK_CYCLING = Beta('B_DEPARTURE_INTER_PEAK_CYCLING', 0, None, None, 0)
        # B_DEPARTURE_INTER_PEAK_PT = Beta('B_DEPARTURE_INTER_PEAK_PT',0,-10,10,0)
        B_DEPARTURE_INTER_PEAK_DRIVING = Beta('B_DEPARTURE_INTER_PEAK_DRIVING', 0, None, None, 0)

        # New variables
        self._new_variables(database_train)

        # Utility functions

        V1 = (#ASC_WALKING +
              B_TIME_WALKING * dur_walking
              #B_PURPOSE_B_WALKING * purpose_B +
              #B_PURPOSE_HBW_WALKING * purpose_HBW +
            #   B_PURPOSE_HBE_WALKING * purpose_HBE +
            #   B_PURPOSE_HBO_WALKING * purpose_HBO +
            #   B_VEHICLE_OWNERSHIP_1_WALKING * co1 +
            #   B_VEHICLE_OWNERSHIP_2_WALKING * co2 +
            #   B_FEMALE_WALKING * female +
            #   B_WINTER_WALKING * winter +
            #   B_AGE_CHILD_WALKING * child +
            #   B_AGE_PENSIONER_WALKING * pensioner +
            #   B_DRIVING_LICENCE_WALKING * driving_license +
            #   B_DAY_WEEK_WALKING * weekday +
            #   B_DAY_SAT_WALKING * saturday +
              # B_DEPARTURE_AM_PEAK_WALKING * ampeak +
            #   B_DEPARTURE_INTER_PEAK_WALKING * interpeak +
            #   B_DEPARTURE_PM_PEAK_WALKING * pmpeak +
            #   B_DISTANCE_WALKING * distance_km
              )

        V2 = (ASC_CYCLING +
              B_TIME_CYCLING * dur_cycling +
              B_PURPOSE_B_CYCLING * purpose_B +
              B_PURPOSE_HBW_CYCLING * purpose_HBW +
              # B_PURPOSE_HBE_CYCLING * purpose_HBE +
              B_PURPOSE_HBO_CYCLING * purpose_HBO +
              B_VEHICLE_OWNERSHIP_CYCLING * co1 +
              B_VEHICLE_OWNERSHIP_CYCLING * co2 +
              B_FEMALE_CYCLING * female +
              B_WINTER_CYCLING * winter +
              # B_AGE_CHILD_CYCLING * child +
              B_AGE_PENSIONER_CYCLING * pensioner +
              B_DRIVING_LICENCE_CYCLING * driving_license +
              # B_DAY_WEEK_CYCLING * weekday +
              B_DAY_SAT_CYCLING * saturday +
              # B_DEPARTURE_AM_PEAK_CYCLING * ampeak +
              B_DEPARTURE_INTER_PEAK_CYCLING * interpeak +
              B_DEPARTURE_PM_PEAK_CYCLING * pmpeak +
              B_DISTANCE_CYCLING * distance_km
              )

        V3 = (ASC_PT +
              B_COST_PT * cost_transit +
              B_TIME_PT_ACCESS * dur_pt_access +
              B_TIME_PT_RAIL * dur_pt_rail +
              B_TIME_PT_BUS * dur_pt_bus +
              B_TIME_PT_INT_WAIT * dur_pt_int_waiting +
              B_TIME_PT_INT_WALK * dur_pt_int_walking +
              B_PURPOSE_B_PT * purpose_B +
              B_PURPOSE_HBW_PT * purpose_HBW +
              B_PURPOSE_HBE_PT * purpose_HBE +
              B_PURPOSE_HBO_PT * purpose_HBO +
              B_VEHICLE_OWNERSHIP_1_PT * co1 +
              B_VEHICLE_OWNERSHIP_2_PT * co2 +
              B_FEMALE_PT * female +
              # B_WINTER_PT * winter +
              B_AGE_CHILD_PT * child +
              B_AGE_PENSIONER_PT * pensioner +
              B_DRIVING_LICENCE_PT * driving_license +
              B_DAY_WEEK_PT * weekday +
              B_DAY_SAT_PT * saturday +
              # B_DEPARTURE_AM_PEAK_PT * ampeak +
              # B_DEPARTURE_INTER_PEAK_PT * interpeak +
              B_DEPARTURE_PM_PEAK_PT * pmpeak +
              B_DISTANCE_PT * distance_km
              )

        V4 = (ASC_DRIVING +
              B_TIME_DRIVING * dur_driving +
              B_COST_DRIVE * drive_cost +
              B_TRAFFIC_DRIVING * driving_traffic_percent +
              B_PURPOSE_B_DRIVING * purpose_B +
              B_PURPOSE_HBW_DRIVING * purpose_HBW +
              B_PURPOSE_HBE_DRIVING * purpose_HBE +
              # B_PURPOSE_HBO_DRIVING * purpose_HBO +
              B_VEHICLE_OWNERSHIP_1_DRIVING * co1 +
              B_VEHICLE_OWNERSHIP_2_DRIVING * co2 +
              B_FEMALE_DRIVING * female +
              B_WINTER_DRIVING * winter +
              B_AGE_CHILD_DRIVING * child +
              B_AGE_PENSIONER_DRIVING * pensioner +
              B_DRIVING_LICENCE_DRIVING * driving_license +
              B_DAY_WEEK_DRIVING * weekday +
              # B_DAY_SAT_DRIVING * saturday +
              # B_DEPARTURE_AM_PEAK_DRIVING * ampeak +
              B_DEPARTURE_INTER_PEAK_DRIVING * interpeak +
              B_DEPARTURE_PM_PEAK_DRIVING * pmpeak +
              B_DISTANCE_DRIVING * distance_km
              )

        # Associate utility functions with the numbering of alternatives
        V = {0: V1,
             1: V2,
             2: V3,
             3: V4}

        av = {0: 1,
              1: 1,
              2: 1,
              3: 1}
        
        # Definition of the model. This is the contribution of each
        # observation to the log likelihood function.
        logprob = loglogit(V, av, choice)

        # Create the Biogeme object
        biogeme = bio.BIOGEME(database_train, logprob)
        biogeme.modelName = 'Biogeme-model'

        biogeme.generate_html = False
        biogeme.generate_pickle = True

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
        
    def _bio_to_rumboost(self, all_columns = False, monotonic_constraints = True, interaction_contraints = True):
        '''
        Converts a biogeme model to a rumboost dict
        '''
        utils = self.model.loglike.util
        rum_structure = []
        
        for k, v in utils.items():
            rum_structure.append({'columns': [], 'monotone_constraints': [], 'interaction_constraints': [], 'betas': [], 'categorical_feature': []})
            for i, pair in enumerate(self._process_parent(v, [])):
                rum_structure[-1]['columns'].append(pair[1])
                rum_structure[-1]['betas'].append(pair[0])
                if interaction_contraints:
                    rum_structure[-1]['interaction_constraints'].append([i])
                if ('TIME' not in pair[0]) & ('COST' not in pair[0]) & ('DISTANCE' not in pair[0]) & ('TRAFFIC' not in pair[0]):
                    rum_structure[-1]['categorical_feature'].append(i)
                bounds = self.model.getBoundsOnBeta(pair[0])
                if monotonic_constraints:
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
            if all_columns:
                rum_structure[-1]['columns'] = [col for col in self.model.database.data.columns.values.tolist() if col != 'choice']
        return rum_structure

    def bio_rum_train(self, valid_test=False, with_pw = False, lr = 0.1, md=1, all_columns = False, monotonic_constraints = True, interaction_constraints = True, save_model = True):
        rum_structure = self._bio_to_rumboost(all_columns=all_columns, monotonic_constraints=monotonic_constraints, interaction_contraints=interaction_constraints)
        
        self.params['learning_rate'] = lr
        self.params['max_depth'] = md
        self.params['early_stopping_rounds'] = 50
        self.params['num_boost_round'] = 3000
        self.params['boosting'] = 'gbdt'
        self.params['monotone_constraints_method'] =  'advanced'
        self.params['min_sum_hessian'] = 1e-6
        self.params['min_data_in_leaf'] = 1
        
        # self.params['bagging_fraction'] = 0.7
        # self.params['feature_fraction'] = 0.7
        # self.params['feature_fraction_bynode'] = 0.7
        # self.params['max_delta_step'] = 1
        # self.params['lambda_l1'] = 0.006
        # self.params['lambda_l2'] = 2

        data = self.model.database.data
        target = self.model.loglike.choice.name
        train_data = lgb.Dataset(data, label=data[target], free_raw_data=False)
        
        if not valid_test:
            model_rumtrained = rum_train(self.params, train_data, valid_sets=[train_data], rum_structure=rum_structure, pw_utility=with_pw)
        else:
            bio_validate_data = self._bio_database(self.dataset_test, 'LTDS_test')
            self._new_variables(bio_validate_data)
            validate_data = lgb.Dataset(bio_validate_data.data, label=bio_validate_data.data[target], free_raw_data=False)

            model_rumtrained = rum_train(self.params, train_data, valid_sets=[validate_data], rum_structure=rum_structure, pw_utility=with_pw)

        self.gbru_cross_entropy = model_rumtrained.best_score
        self.gbru_model = model_rumtrained

        if save_model:
            self.gbru_model.save_model('LTDS_54_gbru_model_{}_depth{}_pw{}_mono{}_interac{}.json'.format(self.params['learning_rate'], md, with_pw, monotonic_constraints, interaction_constraints))

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
        biogeme.modelName = "LTDS_logit_test"

        betaValues = self.betas

        self.bio_prediction = biogeme.simulate(betaValues)

        target = self.model.loglike.choice.name

        bioce_test = 0
        for i,l in enumerate(self.dataset_test[target]):
            bioce_test += np.log(self.bio_prediction.iloc[i,l])
        self.bio_cross_entropy_test = -bioce_test/len(self.dataset_test[target])

    def _rum_predict(self, data_test = None):
        '''
        predictions on the test set from the GBRU model
        '''
        if data_test is None:
            data_test = self.dataset_test
        target = self.model.loglike.choice.name
        features = [f for f in data_test.columns if f != target]
        test_data = lgb.Dataset(data_test.loc[:, features], label=data_test[[target]], free_raw_data=False)
        self.gbru_prediction = self.gbru_model.predict(test_data)
        test_data.construct()
        self.gbru_cross_entropy_test = self.gbru_model.cross_entropy(self.gbru_prediction,test_data.get_label().astype(int))
        self.gbru_accuracy_test = self.gbru_model.accuracy(self.gbru_prediction,test_data.get_label().astype(int))

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

