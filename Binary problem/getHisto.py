import numpy as np
import pandas as pd
import biogeme
import sklearn

ned_data_rp = pd.read_table("Data/netherlandsRP.dat")

X = ned_data_rp[['car_cost', 'rail_cost', 'car_ivtt', 'car_walk_time', 'rail_ivtt', 'rail_acc_time', 'rail_egr_time']]
X_updt = X.copy()
X_updt['car_tt'] = X['car_ivtt'] +X['car_walk_time']
X_updt['rail_tt'] = X['rail_ivtt'] +X['rail_acc_time'] +X['rail_egr_time']

rate_G2E = 0.44378022
X_updt['car_cost'] = X_updt['car_cost'] * rate_G2E
X_updt['rail_cost'] = X_updt['rail_cost'] * rate_G2E
X_updt = X_updt.drop(columns=['car_ivtt', 'car_walk_time', 'rail_ivtt', 'rail_acc_time', 'rail_egr_time'])

y = ned_data_rp['choice']

from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split(X_updt, y, test_size=0.20, random_state=42)

import xgboost as xgb 

xgb_constrained = xgb.XGBClassifier(#objective='binary:logistic', 
                                    #learning_rate = 0.2,
                                    n_estimators = 5, 
                                    max_depth = 1, 
                                    random_state = 42,
                                    monotone_constraints = (-1, -1, -1, -1),
                                    interaction_constraints = [["car_cost"], ["rail_cost"], ["car_tt"], ["rail_tt"]]
                                   )

eval_set = [(X_validate, y_validate)]
xgb_constrained.fit(X_train, y_train, eval_set=eval_set)

y_pred=xgb_constrained.predict(X_validate)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_validate, y_pred))

from xgbfir_modified import getHistoData

histoData = getHistoData(xgb_constrained)