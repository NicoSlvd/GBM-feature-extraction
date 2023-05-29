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

class ltds():

    def __init__(self, model_file = None):
        '''
        Class for the model related to LTDS

        ----------
        parameters

        model_file: str
            file path to load a gbru model already saved
        '''
        self.dataset_path_train = 'Data/LPMC_train.csv'
        self.dataset_path_test = 'Data/LPMC_test.csv'
        self.dataset_name = 'LTDS'
        self.test_size = 0.2
        self.random_state = 42

        self.params = {'num_boost_round': 3000,
                       'verbosity': 1,
                       'objective':'multiclass',
                       'num_classes': 4,
                       'early_stopping_rounds': 50,
                       'boosting': 'gbdt',
                       'monotone_constraints_method': 'advanced'
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

    def _model(self):
        '''
        Create a MNL on the LTDS dataset
        Source of the model: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice
        '''
        database_train = db.Database('LTDS_train', self.dataset_train)

        logger = blog.get_screen_logger(level=blog.DEBUG)
        logger.info('Model LTDS.py')

        globals().update(database_train.variables)
        MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
        #MNL_beta_params_negative = ['B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car']
        MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car']
        #MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_distance_Walk', 'B_distance_Bike', 'B_female_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']
        MNL_beta_params_neutral = ['B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_distance_Walk', 'B_distance_Bike', 'B_female_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']
        # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']

        # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car']

        # MNL_beta_params_neutral = ['B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_distance_Walk', 'B_distance_Bike', 'B_female_Car', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']

        MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
                         1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
                         2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
                         3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}
        # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBO_Walk*purpose_HBO + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
        #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
        #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_cost_transit_Public_Transport*cost_transit',
        #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Petrol_Car*fueltype_Petrol + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}
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

    def bio_rum_train(self, valid_test=False, with_pw = False, lr = 0.1, md = 1, all_columns = False, monotonic_constraints = True, interaction_constraints = True, save_model = True):
        rum_structure = self._bio_to_rumboost(all_columns=all_columns, monotonic_constraints=monotonic_constraints, interaction_contraints=interaction_constraints)

        self.params['learning_rate'] = lr
        self.params['max_depth'] = md
        
        # self.params['bagging_fraction'] = 0.935
        # self.params['feature_fraction'] = 0.679
        # self.params['feature_fraction_bynode'] = 0.629

        # self.params['lambda_l1'] = 0.003
        # self.params['lambda_l2'] = 0.0005

        # self.params['min_gain_to_split'] = 4.137
        # self.params['min_sum_hessian'] = 32
        # self.params['min_data_in_leaf'] = 1
        # self.params['max_delta_step'] = 4

        data = self.model.database.data
        target = self.model.loglike.choice.name
        train_data = lgb.Dataset(data, label=data[target], free_raw_data=False)
        validate_data = lgb.Dataset(self.dataset_test, label=self.dataset_test[target], free_raw_data=False)
        if not valid_test:
            model_rumtrained = rum_train(self.params, train_data, valid_sets=[train_data], rum_structure=rum_structure, pw_utility=with_pw)
        else:
            model_rumtrained = rum_train(self.params, train_data, valid_sets=[validate_data], rum_structure=rum_structure, pw_utility=with_pw)

        self.gbru_cross_entropy = model_rumtrained.best_score
        self.gbru_model = model_rumtrained

        if save_model:
            self.gbru_model.save_model('LTDS_gbru_model_{}_depth{}_pw{}_mono{}_interac{}.json'.format(self.params['learning_rate'], md, with_pw, monotonic_constraints, interaction_constraints))

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

    def _rum_predict(self, piece_wise = False):
        '''
        predictions on the test set from the GBRU model
        '''
        target = self.model.loglike.choice.name
        features = [f for f in self.dataset_test.columns if f != target]
        test_data = lgb.Dataset(self.dataset_test.loc[:, features], label=self.dataset_test[[target]], free_raw_data=False)
        test_data.construct()
        self.gbru_prediction = self.gbru_model.predict(test_data, piece_wise=piece_wise)
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

