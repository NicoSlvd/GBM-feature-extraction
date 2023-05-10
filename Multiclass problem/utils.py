import pandas as pd
from sklearn.model_selection import train_test_split
import biogeme.database as db

from biogeme_model import SwissMetro, estimate_model
from biogeme_to_rumbooster import bio_rum_train

def load_prep_data(data_name = None, train_split = False, split_perc = 0.2, seed = 42):
    '''
    load dataset with specified name from the Data folder and split it for training if specified.

    Parameters
    ----------

    data_name: str
        Full name of the dataset, with the extension
    train_split: bool (default = False)
        If true, the dataset will be split according to split_perc 
    split_perc: float (default = 0.2)
        Size of the validation sample
    seed: int (default = 42)
        Seed of the random generator for reproducibility

    Return
    ------
    df or df_train, df_test: DataFrame
        dataset, splitted or not, ready for training
    '''
    #load data
    if '.dat' in data_name:
        df = pd.read_csv('Data/'+data_name, sep='\t')
    elif '.csv' in data_name:
        df = pd.read_csv('Data/'+data_name)
    else:
        raise ValueError('Dataset without .dat or .csv extensions are not supported')
    
    if 'swissmetro' in data_name:
        #keep relevant data for swissmetro
        keep = ((df['PURPOSE']!=1)*(df['PURPOSE']!=3)+(df['CHOICE']==0)) == 0
        df = df[keep]

    #split data if necessary
    if train_split:
        return train_test_split(df, test_size=split_perc, random_state=seed)
    else:
        return df
    
def compare_models(dataset_names):
    '''
    compare one or several models estimated through biogeme and trained through GBRU, by calculating
    the cross-entropy on the train set.

    ----------
    parameters

    dataset_name: list(str)
        list of dataset names (as str)
    '''

    for dataset in dataset_names:
        data_train, _ = load_prep_data(dataset, train_split=True)
        if 'swissmetro' in dataset:
            model = SwissMetro(data_train)
            param = {'max_depth': 1, 
                     'num_boost_round': 300, 
                     'objective':'multiclass',
                     'monotone_constraints': [-1, -1, -1, -1, -1, -1, -1, -1], 
                     'interaction_constraints': [[0], [1], [2], [3], [4], [5], [6], [7]],
                     'learning_rate': 0.2,
                     'verbosity': 2,
                     'num_classes': 3
                    }
        elif 'london' in dataset:
            #model = London(data_train)
            param = {'max_depth': 3, 
                     'num_boost_round': 300, 
                     'objective':'multiclass',
                     'monotone_constraints': [-1, -1, -1, -1, -1, -1], 
                     'interaction_constraints': [[0], [1], [2], [3], [4], [5]],
                     'learning_rate': 0.2,
                     'verbosity': 2,
                     'num_classes': 4
                    }
        else:
            raise ValueError('This dataset is not implemented yet')
        
        bio_model = estimate_model(model)

        gbru_model = bio_rum_train(model, param)

        print('On {}, biogeme has a CE of: {}'.format(dataset, bio_model))
        print('On {}, GBRU has a CE of: {}'.format(dataset, gbru_model))
        