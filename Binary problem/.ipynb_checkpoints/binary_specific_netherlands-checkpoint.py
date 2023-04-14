# Translated to .py by Yundi Zhang
# Jan 2017
# Adapted to PandasBiogeme by Michel Bierlaire
# Sun Oct 21 23:00:22 2018

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, DefineVariable
from biogeme.models import loglogit

pandas = pd.read_table("Data/netherlands.dat")
database = db.Database("netherlands",pandas)
pd.options.display.float_format = '{:.3g}'.format

globals().update(database.variables)

exclude = sp != 0
database.remove(exclude)

# Parameters to be estimated
# Arguments:
#   1  Name for report. Typically, the same as the variable
#   2  Starting value
#   3  Lower bound
#   4  Upper bound
#   5  0: estimate the parameter, 1: keep it fixed
ASC_CAR		 = Beta('ASC_CAR',0,None,None,0)
ASC_RAIL	 = Beta('ASC_RAIL',0,None,None,1)
BETA_COST	 = Beta('BETA_COST',0,None,None,0)
BETA_TT      = Beta('BETA_TT',0,None,None,0)
#BETA_TT_CAR	 = Beta('BETA_TT_CAR',0,None,None,0)
#BETA_TT_RAIL = Beta('BETA_TT_RAIL',0,None,None,0)

# Define here arithmetic expressions for name that are not directly available from the data
rail_time  = DefineVariable('rail_time',(  rail_ivtt   +  rail_acc_time   ) +  rail_egr_time  ,database)
car_time  = DefineVariable('car_time', car_ivtt   +  car_walk_time  ,database)
rate_G2E = DefineVariable('rate_G2E', 0.44378022,database)
car_cost_euro = DefineVariable('car_cost_euro', car_cost * rate_G2E,database)
rail_cost_euro = DefineVariable('rail_cost_euro', rail_cost * rate_G2E,database)

# Utilities
__Car = ASC_CAR  + BETA_COST * car_cost_euro + BETA_TT * car_time
__Rail = ASC_RAIL + BETA_COST * rail_cost_euro + BETA_TT * rail_time
__V = {0: __Car,1: __Rail}
__av = {0: 1,1: 1}

# The choice model is a logit, with availability conditions
logprob = loglogit(__V,__av,choice)
biogeme  = bio.BIOGEME(database,logprob)
#biogeme.modelName = "binary_specific_netherlands"
results = biogeme.estimate()
# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)
print(f"Nbr of observations: {database.getNumberOfObservations()}")
print(f"LL(0) =    {results.data.initLogLike:.3f}")
print(f"LL(beta) = {results.data.logLike:.3f}")
print(f"rho bar square = {results.data.rhoBarSquare:.3g}")
print(f"Output file: {results.data.htmlFileName}")
