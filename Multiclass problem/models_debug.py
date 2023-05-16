from NTS import nts
from LTDS import ltds
import numpy as np

ltds_model = ltds(model_file='LTDS_gbru_model.json')

ltds_model.gbru_model.pw_utility(ltds_model.dataset_train, ltds_model.dataset_test)

#print('Best performance of the model with a learning rate of 0.1: \n------GMPCA: {}\n---Accuracy: {}' \
#      .format(np.exp(-ltds_model.gbru_cross_entropy_test)*100, ltds_model.gbru_accuracy_test*100))