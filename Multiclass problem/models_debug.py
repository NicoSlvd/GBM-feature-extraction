from NTS import nts
import json


nts_model = nts(model_file='nts_gbru_model.json')

print(nts_model.gbru_model.plot_parameters(nts_model.params, nts_model.dataset_train, []))