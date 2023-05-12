import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, DefineVariable
from biogeme.models import loglogit
from sklearn.model_selection import train_test_split

def estimate_model(model, return_params = False):
    '''
    estimate a biogeme model from the biogeme.biogeme object

    ----------
    parameters

    model: biogeme.biogeme
        biogeme model, currently has to be a MNL model
    return_params: bool (default = False)
        if True, returns parameters in a DataFrame as well

    ------
    return
    
    cr_entropy (, pandasResults): float (, DataFrame)
        cross entropy after estimation and - if specified - parameters of the model
    '''
    #estimate model
    results = model.estimate()

    #results
    pandasResults = results.getEstimatedParameters()
    print(pandasResults)
    print(f"Nbr of observations: {model.database.getNumberOfObservations()}")
    print(f"LL(0) = {results.data.initLogLike:.3f}")
    print(f"LL(beta) = {results.data.logLike:.3f}")
    print(f"rho bar square = {results.data.rhoBarSquare:.3g}")
    print(f"Output file: {results.data.htmlFileName}")

    #cross entropy
    cr_entropy = -results.data.logLike / model.database.getNumberOfObservations()

    if return_params:
        return cr_entropy, pandasResults
    else:
        return cr_entropy

def SwissMetro(df):
    '''
    create a MNL on the swissmetro dataset

    ----------
    parameters

    df: DataFrame
        Database used to train the model

    ------
    return

    biogeme: Biogeme.biogeme
        biogeme model, ready to be trained

    '''
    database_train = db.Database('swissmetro_train', df)

    globals().update(database_train.variables)

    # Parameters to be estimated
    ASC_CAR   = Beta('ASC_CAR', 0, None, None, 0)
    ASC_SM    = Beta('ASC_SM',  0, None, None, 0)
    ASC_TRAIN = Beta('ASC_SBB', 0, None, None, 1)

    B_TIME = Beta('B_TIME', 0, None, 0, 0)
    B_COST = Beta('B_COST', 0, None, 0, 0)
    B_HE   = Beta('B_HE',   0, None, 0, 0)

    # Definition of new variables
    TRAIN_COST = database_train.DefineVariable('TRAIN_COST', TRAIN_CO * ( GA == 0 ))
    SM_COST    = database_train.DefineVariable('SM_COST', SM_CO * ( GA == 0 ))

    # Utilities
    V_TRAIN = ASC_TRAIN + B_TIME * TRAIN_TT + B_COST * TRAIN_COST + B_HE * TRAIN_HE
    V_SM    = ASC_SM    + B_TIME * SM_TT    + B_COST * SM_COST    + B_HE * SM_HE
    V_CAR   = ASC_CAR   + B_TIME * CAR_TT   + B_COST * CAR_CO

    V = {1: V_TRAIN, 2: V_SM, 3: V_CAR}
    av = {1: TRAIN_AV, 2: SM_AV, 3: CAR_AV}

    # Choice model estimation
    logprob = loglogit(V, av, CHOICE)
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "SwissmetroMNL"

    biogeme.generate_html = False
    biogeme.generate_pickle = False

    return biogeme