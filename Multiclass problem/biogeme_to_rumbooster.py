import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta
from biogeme.models import loglogit
import lightgbm as lgb
from rumbooster import rum_train

def process_parent(parent, pairs):
    if parent.getClassName() == 'Times':
        pairs.append(get_pair(parent))
    else:
        try:
            left = parent.left
            right = parent.right
        except:
            return pairs
        else:
            process_parent(left, pairs)
            process_parent(right, pairs)
        return pairs
    
def get_pair(parent):
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
        
def bio_to_rumboost(biogeme_model):
    '''
    Converts a biogeme model to a rumboost dict
    '''
    utils = biogeme_model.loglike.util
    rum_structure = []
    
    for k, v in utils.items():
        rum_structure.append({'columns': [], 'monotone_constraints': [], 'interaction_constraints': [], 'betas': []})
        for i, pair in enumerate(process_parent(v, [])):
            rum_structure[-1]['columns'].append(pair[1])
            rum_structure[-1]['betas'].append(pair[0])
            rum_structure[-1]['interaction_constraints'].append([i])
            bounds = biogeme_model.getBoundsOnBeta(pair[0])
            if (bounds[0] is None) and (bounds[1] is None):
                raise ValueError("Only one bound can be not None")
            if bounds[0] is not None:
                if bounds[0] >= 0:
                    rum_structure[-1]['monotone_constraints'].append(1)
            elif bounds[1] is not None:
                if bounds[1] <= 0:
                    rum_structure[-1]['monotone_constraints'].append(-1)
            else:
                rum_structure[k]['monotone_constraints'].append(0)
    return rum_structure

def bio_rum_train(biogeme_model, param, ret_train_model = False):
    rum_structure = bio_to_rumboost(biogeme_model)
    data = biogeme_model.database.data
    target = biogeme_model.loglike.choice.name
    train_data = lgb.Dataset(data, label=data[target]-1, free_raw_data=False)
    model_rumtrained = rum_train(param, train_data, valid_sets=[train_data], rum_structure=rum_structure)

    if ret_train_model:
        return model_rumtrained.best_score, model_rumtrained
    else:
        return model_rumtrained.best_score