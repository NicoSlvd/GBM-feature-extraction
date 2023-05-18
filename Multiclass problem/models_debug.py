from NTS import nts
from LTDS_Tim import ltds_54
from LTDS import ltds
import numpy as np

ltds_model = ltds()
#ltds_model = ltds_54()

ltds_model.bio_rum_train(with_pw=True)
#ltds_model.gbru_model.plot_parameters(ltds_model.params, ltds_model.dataset_train, utility_names = {'0': 'Walking', '1': 'Cycling', '2': 'Public transport','3': 'Driving'}, with_pw = True, with_stairs=True)

#pw_preds = ltds_model.gbru_model.pw_predict(ltds_model.dataset_train)

#print(ltds_model.gbru_model.cross_entropy(pw_preds, ltds_model.dataset_train.choice))

#print('Best performance of the model with a learning rate of 0.1: \n------GMPCA: {}\n---Accuracy: {}' \
#      .format(np.exp(-ltds_model.gbru_cross_entropy_test)*100, ltds_model.gbru_accuracy_test*100))