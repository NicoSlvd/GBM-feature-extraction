import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split

import sys
sys.path.append("LightGBM/python-package/lightgbm")

swissmetro = pd.read_table('Data/swissmetro.dat')
swissmetro = swissmetro[swissmetro['CHOICE']!=0]

new_train_co = swissmetro['TRAIN_CO'] * (swissmetro['GA']==0)
new_sm_co = swissmetro['SM_CO'] * (swissmetro['GA']==0)

feature = swissmetro[['TRAIN_TT', 'TRAIN_CO', 'TRAIN_HE', 'SM_TT', 'SM_CO', 'SM_HE', 'CAR_TT', 'CAR_CO']]

feature['TRAIN_CO'] = new_train_co
feature['SM_CO'] = new_sm_co

choice = swissmetro[['CHOICE']] -1
X_train, X_test, y_train, y_test = train_test_split(feature, choice, test_size=0.2, random_state = 42)

train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test, reference= train_data)

param = {'max_depth': 1, 
         'objective': 'multiclass', 
         'num_iterations': 300, 
         'monotone_constraints': [-1, -1, -1, -1, -1, -1, -1, -1], 
         'interaction_constraints': [[0], [1], [2], [3], [4], [5], [6], [7]],
         'learning_rate': 0.3,
         'verbosity': 1,
         'num_classes': 3
        }

lightgbm_1 = lgb.train(param, train_data, valid_sets=[validation_data])