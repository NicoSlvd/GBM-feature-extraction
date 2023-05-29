from NTS import nts
from LTDS_Tim import ltds_54
from LTDS import ltds
import numpy as np
import json

#ltds_54_model = ltds_54(model_file='LTDS_54_gbru_model_0.05_depth1_pwFalse_monoTrue_interacTrue.json')

ltds_model = ltds()





#a = ltds_54_model.gbru_model.plot_parameters(ltds_54_model.params, ltds_54_model.dataset_train, utility_names = {'0': 'Walking', '1': 'Cycling', '2': 'Public transport','3': 'Driving'}, save_figure=True)
#ltds_model = ltds(model_file='LTDS_gbru_model_0.2_pwFalse.json')

ltds_model.bio_rum_train(valid_test=True, with_pw = False, lr = 0.2, md = 1, all_columns=False, interaction_constraints=True, monotonic_constraints=True, save_model=False)
#b=ltds_model.gbru_model.plot_parameters(ltds_model.params, ltds_model.dataset_train, utility_names = {'0': 'Walking', '1': 'Cycling', '2': 'Public transport','3': 'Driving'}, save_figure=True)
# #ltds_54_model.bio_rum_train(valid_test=True, lr = 0.1, md = 1, all_columns = False, interaction_constraints=False, monotonic_constraints=True, save_model=False)

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