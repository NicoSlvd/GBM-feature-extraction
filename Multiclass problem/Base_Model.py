# Translated to .py by Meritxell Pacheco (December 2016)
# Adapted to PandasBiogeme by Nicola Ortelli (November 2019)


import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, DefineVariable
from biogeme.models import loglogit
import biogeme.models as models
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data/swissmetro.dat", sep = '\t')
database = db.Database("swissmetro", df)

# Exclude data
exclude = (( PURPOSE != 1 ) * ( PURPOSE != 3 ) + ( CHOICE == 0 )) > 0
database.remove(exclude)

df_train, df_test = train_test_split(df, test_size=0.2, random_state = 42)

database_train = db.Database("swissmetro_train", df_train)
database_test = db.Database("swissmetro_test", df_test)

# Parameters to be estimated
ASC_CAR   = Beta('ASC_CAR', 0, None, None, 0)
ASC_SM    = Beta('ASC_SM',  0, None, None, 0)
ASC_TRAIN = Beta('ASC_SBB', 0, None, None, 1)

B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
B_HE   = Beta('B_HE',   0, None, None, 0)

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
biogeme.modelName = "Base Model"

biogeme.generateHtml = False
biogeme.generatePickle = False

results = biogeme.estimate()

# Results
pandasResults = results.getEstimatedParameters()
print(pandasResults)
print(f"Nbr of observations: {database_train.getNumberOfObservations()}")
print(f"LL(0) = {results.data.initLogLike:.3f}")
print(f"LL(beta) = {results.data.logLike:.3f}")
print(f"rho bar square = {results.data.rhoBarSquare:.3g}")
print(f"Output file: {results.data.htmlFileName}")