import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm.rumbooster import rum_train, rum_cv

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
         'num_iterations': 1000, 
         'objective':'multiclass',
         'monotone_constraints': [-1, -1, -1, -1, -1, -1, -1, -1], 
         'interaction_constraints': [[0], [1], [2], [3], [4], [5], [6], [7]],
         'learning_rate': 0.3,
         'verbosity': 1,
         'num_classes': 3
         #'early_stopping_round': 5
        }

train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
validation_data = lgb.Dataset(X_test, label=y_test, reference= train_data, free_raw_data=False)

rum_structure_1= [{'columns': ['TRAIN_TT', 'TRAIN_CO', 'TRAIN_HE'], 
                  'monotone_constraints': [-1, -1, -1],
                  'interaction_constraints': [0, 1, 2]}, 
                 {'columns': ['SM_TT', 'SM_CO', 'SM_HE'], 
                  'monotone_constraints': [-1, -1, -1],
                  'interaction_constraints': [0, 1, 2]},
                 {'columns': ['CAR_TT', 'CAR_CO'], 
                  'monotone_constraints': [-1, -1],
                  'interaction_constraints': [0, 1]}]

lightgbm_1 = rum_train(param, train_data, valid_sets=[train_data], rum_structure=rum_structure_1)
'''lightgbm_cv = rum_cv(param, train_data, num_boost_round=10,
                     folds=None, nfold=5, stratified=True, shuffle=True,
                     metrics=None, fobj=None, feval=None, init_model=None,
                     feature_name='auto', categorical_feature='auto',
                     early_stopping_rounds=None, fpreproc=None,
                     verbose_eval=True, show_stdv=True, seed=0,
                     callbacks=None, eval_train_metric=False,
                     return_cvbooster=False, rum_structure=rum_structure_1)'''
                     
#lightgbm_1 = rum_train(param, train_data, rum_structure= rum_structure_1)

#lightgbm_1.plot_parameters(param, X_train, ['h', 'h', 'h', 'h', 'h', 'h', 'h', 'h'])