# Translated to .py by Yundi Zhang
# Jan 2017
# Adapted to PandasBiogeme by Michel Bierlaire
# Sun Oct 21 23:00:22 2018

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, DefineVariable
from biogeme.models import loglogit
import biogeme.models as models

pandas = pd.read_table("Data/netherlands.dat")
#database = db.Database("netherlands",pandas)
df_train, df_test = train_test_split(pandas, test_size=0.2, random_state = 42)

database_train = db.Database("netherlands_train", df_train)
database_test = db.Database("netherlands_test", df_test)
pd.options.display.float_format = '{:.3g}'.format

globals().update(database_train.variables)

exclude = sp != 0
#database.remove(exclude)
database_train.remove(exclude)
database_test.remove(exclude)

# Parameters to be estimated
# Arguments:
#   1  Name for report. Typically, the same as the variable
#   2  Starting value
#   3  Lower bound
#   4  Upper bound
#   5  0: estimate the parameter, 1: keep it fixed
ASC_CAR	       = Beta('ASC_CAR',0,None,None,0)
ASC_RAIL	   = Beta('ASC_RAIL',0,None,None,1)
BETA_COST_CAR  = Beta('BETA_COST_CAR',0,None,None,0)
BETA_COST_RAIL = Beta('BETA_COST_RAIL',0,None,None,0)
BETA_TT_CAR    = Beta('BETA_TT_CAR',0,None,None,0)
BETA_TT_RAIL   = Beta('BETA_TT_RAIL',0,None,None,0)
#BETA_TT_CAR	 = Beta('BETA_TT_CAR',0,None,None,0)
#BETA_TT_RAIL = Beta('BETA_TT_RAIL',0,None,None,0)

# Define here arithmetic expressions for name that are not directly available from the data
rail_time  = DefineVariable('rail_time',(  rail_ivtt   +  rail_acc_time   ) +  rail_egr_time  ,database_train)
car_time  = DefineVariable('car_time', car_ivtt   +  car_walk_time  ,database_train)
rate_G2E = DefineVariable('rate_G2E', 0.44378022,database_train)
car_cost_euro = DefineVariable('car_cost_euro', car_cost * rate_G2E,database_train)
rail_cost_euro = DefineVariable('rail_cost_euro', rail_cost * rate_G2E,database_train)

rail_time  = DefineVariable('rail_time',(  rail_ivtt   +  rail_acc_time   ) +  rail_egr_time  ,database_test)
car_time  = DefineVariable('car_time', car_ivtt   +  car_walk_time  ,database_test)
rate_G2E = DefineVariable('rate_G2E', 0.44378022,database_test)
car_cost_euro = DefineVariable('car_cost_euro', car_cost * rate_G2E,database_test)
rail_cost_euro = DefineVariable('rail_cost_euro', rail_cost * rate_G2E,database_test)

# Utilities
__Car = ASC_CAR  + BETA_COST_CAR * car_cost_euro + BETA_TT_CAR * car_time
__Rail = ASC_RAIL + BETA_COST_RAIL * rail_cost_euro + BETA_TT_RAIL * rail_time
__V = {0: __Car,1: __Rail}
__av = {0: 1,1: 1}

# The choice model is a logit, with availability conditions
logprob = loglogit(__V,__av,choice)
biogeme  = bio.BIOGEME(database_train,logprob)
biogeme.modelName = "binary_specific_netherlands"
biogeme.generateHtml = False
biogeme.generatePickle = False

results = biogeme.estimate()
# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)
print(f"Nbr of observations: {database_train.getNumberOfObservations()}")
print(f"LL(0) =    {results.data.initLogLike:.3f}")
print(f"LL(beta) = {results.data.logLike:.3f}")
print(f"rho bar square = {results.data.rhoBarSquare:.3g}")
print(f"Output file: {results.data.htmlFileName}")

prob_car = models.logit(__V, __av, 0)
prob_train = models.logit(__V, __av, 1)

simulate ={'Prob. car': prob_car,
           'Prob. train':  prob_train}

biogeme = bio.BIOGEME(database_test, simulate)
biogeme.modelName = "netherland_logit_test"

betas = biogeme.freeBetaNames

betaValues = results.getBetaValues()

simulatedValues = biogeme.simulate(betaValues)

prob_max = simulatedValues.idxmax(axis=1)
prob_max = prob_max.replace({'Prob. car': 0, 'Prob. train': 1})

data = {'y_Actual':    df_test['choice'],
        'y_Predicted': prob_max
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

print(confusion_matrix)

accuracy = np.diagonal(confusion_matrix.to_numpy()).sum()/confusion_matrix.to_numpy().sum()
print('Global accuracy of the model:', accuracy)