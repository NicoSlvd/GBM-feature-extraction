from NTS import nts
from LTDS_Tim import ltds_54
from LTDS import ltds
import numpy as np

# import pandas as pd
# from sklearn.model_selection import train_test_split

# import biogeme.database as db
# import biogeme.biogeme as bio
# from biogeme.expressions import Beta
# from biogeme.models import loglogit
# from sklearn.model_selection import train_test_split

# df = pd.read_csv('Data/swissmetro.dat', sep='\t')
# keep = ((df['PURPOSE']!=1)*(df['PURPOSE']!=3)+(df['CHOICE']==0)) == 0
# df = df[keep]
# df.loc[:, 'TRAIN_COST'] = df['TRAIN_CO'] * (df['GA']==0)
# df.loc[:, 'SM_COST'] = df['SM_CO'] * (df['GA']==0)
# df_final = df[['TRAIN_TT', 'TRAIN_COST', 'TRAIN_HE', 'SM_TT', 'SM_COST', 'SM_HE', 'CAR_TT', 'CAR_CO', 'CHOICE']]
# df_train, df_test  = train_test_split(df_final, test_size=0.2, random_state=2023)
# #ltds_54_model = ltds()

ltds_model = ltds()

# database_train = db.Database('swissmetro_train', df_train)

# globals().update(database_train.variables)

# # Parameters to be estimated
# ASC_CAR   = Beta('ASC_CAR', 0, None, None, 0)
# ASC_SM    = Beta('ASC_SM',  0, None, None, 0)
# ASC_TRAIN = Beta('ASC_SBB', 0, None, None, 1)

# B_TIME = Beta('B_TIME', 0, None, 0, 0)
# B_COST = Beta('B_COST', 0, None, 0, 0)
# B_HE   = Beta('B_HE',   0, None, 0, 0)

# # Utilities
# V_TRAIN = ASC_TRAIN + B_TIME * TRAIN_TT + B_COST * TRAIN_COST + B_HE * TRAIN_HE
# V_SM    = ASC_SM    + B_TIME * SM_TT    + B_COST * SM_COST    + B_HE * SM_HE
# V_CAR   = ASC_CAR   + B_TIME * CAR_TT   + B_COST * CAR_CO

# V = {1: V_TRAIN, 2: V_SM, 3: V_CAR}
# av = {1: 1, 2: 1, 3: 1}

# # Choice model estimation
# logprob = loglogit(V, av, CHOICE)
# biogeme = bio.BIOGEME(database_train, logprob)
# biogeme.modelName = "SwissmetroMNL"

# biogeme.generate_html = False
# biogeme.generate_pickle = False

# from demo_swissmetro import rumb_train

# params = {'max_depth': 2, 
#             'num_boost_round': 100, 
#             'objective':'multiclass',
#             'learning_rate': 0.3,
#             'verbosity': 1,
#             'num_classes': 3,
#             'min_sum_hessian': 1e-6,
#             'min_data_in_leaf': 1,
#             'early_stopping_round':5
#             }

# rumb_demo = rumb_train(biogeme, params)

# print(rumb_demo.getweights_v2())

# rumb_demo.plot_2d('TRAIN_COST','TRAIN_TT', np.max(df_train['TRAIN_TT']),np.max(df_train['TRAIN_COST']))



#ltds_54_model.gbru_model.plot_parameters(ltds_54_model.params, ltds_54_model.dataset_train, utility_names = {'0': 'Walking', '1': 'Cycling', '2': 'Public transport','3': 'Driving'}, save_figure=True)
#ltds_model = ltds(model_file='LTDS_gbru_model_0.2_pwFalse.json')

ltds_model.bio_rum_train(valid_test=True, with_pw = False, lr = 0.2, md = 2, all_columns=False, interaction_constraints=True, monotonic_constraints=True, save_model=False)
#b=ltds_model.gbru_model.plot_parameters(ltds_model.params, ltds_model.dataset_train, utility_names = {'0': 'Walking', '1': 'Cycling', '2': 'Public transport','3': 'Driving'}, save_figure=True)
# #ltds_54_model.bio_rum_train(valid_test=True, lr = 0.1, md = 1, all_columns = False, interaction_constraints=False, monotonic_constraints=True, save_model=False)
ltds_model.gbru_model.getweights_v2()
ltds_model.gbru_model.plot_2d('dur_driving', 'cost_driving_total', 2.2, 18)
# learning_rates = [0.1]

# interaction = [True, False]
# #monotonic = [True, False]
# md = [3, 5]
# best_score1 = 1000
# best_score2 = 1000

# ltds_54_model = ltds_54()
# ltds_54_model.bio_rum_train(valid_test=True, with_pw = False, lr = 0.1, md = 1, interaction_constraints=True, monotonic_constraints=True, save_model=True)


# for lr in learning_rates:
#     for inter in interaction:
#         if not inter:
#             for d in md:
#                 ltds_54_model = ltds_54()
#                 ltds_54_model.bio_rum_train(valid_test=True, lr = lr, md = d, interaction_constraints=inter, monotonic_constraints=True, save_model=True)
#                 bs = ltds_54_model.gbru_model.best_score
#                 if bs < best_score1:
#                     best_score1 = bs
#                     best_lr1 = lr
#                     best_inter1 = inter
#                     best_d = d
#         else:
#             if lr == 0.05:
#                 continue
#             else:
#                 ltds_54_model = ltds_54()
#                 ltds_54_model.bio_rum_train(valid_test=True, lr = lr, md = 1, interaction_constraints=inter, monotonic_constraints=True, save_model=True)
#                 bs = ltds_54_model.gbru_model.best_score
#                 if bs < best_score2:
#                         best_score2 = bs
#                         best_lr2 = lr
#                         best_inter2 = inter


# with open('best_lr1.txt', 'w') as f:
#     json.dump(best_lr1, f)
# with open('best_score1.txt', 'w') as f:
#     json.dump(best_score1, f)
# with open('best_inter1.txt', 'w') as f:
#     json.dump(best_inter1, f)
# with open('best_depth.txt', 'w') as f:
#     json.dump(best_d, f)
# with open('best_lr2.txt', 'w') as f:
#     json.dump(best_lr2, f)
# with open('best_score2.txt', 'w') as f:
#     json.dump(best_score2, f)
# with open('best_inter2.txt', 'w') as f:
#     json.dump(best_inter2, f)


# learning_rates = [0.1, 0.2]
# with_pw = [True, False]
# best_score = 1000

# for lr in learning_rates:
#     for w_pw in with_pw:
#         ltds_model = ltds()
#         ltds_54_model = ltds_54()
#         ltds_model.bio_rum_train(valid_test=True, with_pw=w_pw, lr = lr)
#         bs = ltds_model.gbru_model.best_score
#         if bs < best_score:
#             best_score = bs
#             best_lr = lr
#             best_pw = w_pw
#         ltds_54_model.bio_rum_train(valid_test=True, with_pw=w_pw, lr = lr)
#         bs = ltds_54_model.gbru_model.best_score
#         if bs < best_score:
#             best_score = bs
#             best_lr = lr
#             best_pw = w_pw

# with open('best_lr.txt', 'w') as f:
#     json.dump(best_lr, f)
# with open('best_score.txt', 'w') as f:
#     json.dump(best_score, f)
# with open('best_pw.txt', 'w') as f:
#     json.dump(best_pw, f)

#ltds_model.gbru_model.plot_parameters(ltds_model.params, ltds_model.dataset_train, utility_names = {'0': 'Walking', '1': 'Cycling', '2': 'Public transport','3': 'Driving'}, with_pw = True, with_stairs=True)

#pw_preds = ltds_model.gbru_model.pw_predict(ltds_model.dataset_train)

#print(ltds_model.gbru_model.cross_entropy(pw_preds, ltds_model.dataset_train.choice))

#print('Best performance of the model with a learning rate of 0.1: \n------GMPCA: {}\n---Accuracy: {}' \
#      .format(np.exp(-ltds_model.gbru_cross_entropy_test)*100, ltds_model.gbru_accuracy_test*100))