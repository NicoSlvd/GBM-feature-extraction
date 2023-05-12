import pandas as pd
import numpy as np
import lightgbm as lgb
from rumbooster import rum_train, rum_cv

from sklearn.model_selection import train_test_split

swissmetro = pd.read_table('Data/swissmetro.dat')

keep = (( swissmetro['PURPOSE'] != 1 ) * ( swissmetro['PURPOSE'] != 3 ) + ( swissmetro['CHOICE'] == 0 )) == 0
#swissmetro.drop(swissmetro[exclude].index, inplace=True).reset_index(inplace=True, drop=True)
swissmetro = swissmetro[keep]

new_train_co = swissmetro['TRAIN_CO'] * (swissmetro['GA']==0)
new_sm_co = swissmetro['SM_CO'] * (swissmetro['GA']==0)

feature = swissmetro[['TRAIN_TT', 'TRAIN_CO', 'TRAIN_HE', 'SM_TT', 'SM_CO', 'SM_HE', 'CAR_TT', 'CAR_CO']]

feature['TRAIN_CO'] = new_train_co
feature['SM_CO'] = new_sm_co

choice = swissmetro[['CHOICE']] -1

X_train, X_test, y_train, y_test = train_test_split(feature, choice, test_size=0.2, random_state = 42)

param = {'max_depth': 1, 
         'num_iterations': 30, 
         'objective':'multiclass',
         'monotone_constraints': [-1, -1, -1, -1, -1, -1, -1, -1], 
         'interaction_constraints': [[0], [1], [2], [3], [4], [5], [6], [7]],
         'learning_rate': 0.3,
         'verbosity': 2,
         'num_classes': 3,
         'linear_tree': True
         #'early_stopping_round': 5
        }

train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
validation_data = lgb.Dataset(X_test, label=y_test, reference= train_data, free_raw_data=False, params={
                            'linear_tree': True
                        })

rum_structure_1= [{'columns': ['TRAIN_TT', 'TRAIN_CO', 'TRAIN_HE'], 
                  'monotone_constraints': [-1, -1, -1],
                  'interaction_constraints': [0, 1, 2]}, 
                 {'columns': ['SM_TT', 'SM_CO', 'SM_HE'], 
                  'monotone_constraints': [-1, -1, -1],
                  'interaction_constraints': [0, 1, 2]},
                 {'columns': ['CAR_TT', 'CAR_CO'], 
                  'monotone_constraints': [-1, -1],
                  'interaction_constraints': [0, 1]}]

#lightgbm_1 = rum_train(param, train_data, rum_structure=rum_structure_1)
lightgbm_cv = rum_cv(param, train_data, num_boost_round=10,
                     folds=None, nfold=5, stratified=True, shuffle=True,
                     metrics=None, fobj=None, feval=None, init_model=None,
                     feature_name='auto', categorical_feature='auto',
                     early_stopping_rounds=None, fpreproc=None,
                     verbose_eval=True, show_stdv=True, seed=0,
                     callbacks=None, eval_train_metric=False,
                     return_cvbooster=True, rum_structure=rum_structure_1)


a = 1                    
#lightgbm_1 = rum_train(param, train_data, rum_structure= rum_structure_1)

'''london_data = pd.read_csv("../Binary problem/Data/dataset_london.csv")

london_with_driving_lic = london_data[london_data['driving_license']==1]

london_with_car = london_with_driving_lic[london_with_driving_lic['car_ownership']>0]

ld_with_one_trip = london_with_car.groupby(by=['household_id']).sample(1)

Feature = ld_with_one_trip[['travel_mode', 'survey_year', 'dur_walking', 'dur_cycling', 'dur_pt_total', 'dur_driving', 'cost_transit', 'cost_driving_total']]

X = Feature[['survey_year', 'dur_walking', 'dur_cycling', 'dur_pt_total', 'cost_transit', 'dur_driving', 'cost_driving_total']]
y = Feature[['travel_mode', 'survey_year']]


new_mode = {'drive':0, 'pt':1, 'cycle':2,'walk':3}

y = y.replace({'travel_mode': new_mode})


X_train = X[X['survey_year']<3]
X_validate =  X[X['survey_year']==3]
y_train = y[y['survey_year']<3]
y_validate = y[y['survey_year']==3]

X_train = X_train.drop(columns=['survey_year'])
X_validate = X_validate.drop(columns=['survey_year'])
y_train = y_train.drop(columns=['survey_year'])
y_validate = y_validate.drop(columns=['survey_year'])

param_ld = {'max_depth': 1, 
         'num_boost_round': 450, 
         'objective':'multiclass',
         'monotone_constraints': [-1, -1, -1, -1, -1, -1], 
         'interaction_constraints': [[0], [1], [2], [3], [4], [5]],
         'learning_rate': 0.1,
         'verbosity': 2,
         'num_classes': 4,
         'early_stopping_rounds'
        }

train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
validation_data = lgb.Dataset(X_validate, label=y_validate, reference= train_data, free_raw_data=False)

rum_structure_ld= [{'columns': ['dur_driving', 'cost_driving_total', 'dur_pt_total', 'cost_transit', 'dur_cycling', 'dur_walking'], 
                  'monotone_constraints': [0, 0, 0, 0, 0, 0], 
                  'interaction_constraints': [[0, 1, 2, 3, 4, 5]]}, 
                 {'columns': ['dur_driving', 'cost_driving_total', 'dur_pt_total', 'cost_transit', 'dur_cycling', 'dur_walking'], 
                  'monotone_constraints': [0, 0, 0, 0, 0, 0], 
                  'interaction_constraints': [[0, 1, 2, 3, 4, 5]]},
                 {'columns': ['dur_driving', 'cost_driving_total', 'dur_pt_total', 'cost_transit', 'dur_cycling', 'dur_walking'], 
                  'monotone_constraints': [0, 0, 0, 0, 0, 0], 
                  'interaction_constraints': [[0, 1, 2, 3, 4, 5]]},
                 {'columns': ['dur_driving', 'cost_driving_total', 'dur_pt_total', 'cost_transit', 'dur_cycling', 'dur_walking'], 
                  'monotone_constraints': [0, 0, 0, 0, 0, 0], 
                  'interaction_constraints': [[0, 1, 2, 3, 4, 5]]}]

lightgbm_ld = rum_train(param_ld, train_data, valid_sets=[validation_data], rum_structure= rum_structure_ld)

dict_labels = {'cost_driving_total': 'gbp', 
               'dur_driving':'min', 
               'cost_transit':'gbp', 
               'dur_pt_total':'min', 
               'dur_cycling':'min', 
               'dur_walking':'min'}
#lightgbm_ld.plot_parameters(param_ld, X_train, dict_labels)

param_unc = {'max_depth': 1, 
         'num_boost_round': 450, 
         'objective':'multiclass',
         'learning_rate': 0.1,
         'verbosity': 2,
         'num_classes': 4
        }

lightgbm_1_unc = lgb.train(param_unc, train_data, valid_sets=[validation_data])'''