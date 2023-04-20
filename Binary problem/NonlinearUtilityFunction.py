from xgbfir_modified import getHistoData
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

def arrangeHistoDataForPlot(data):
    
    histoDataForPlot = {}
    for f in data.Feature.unique():
        split_points = []
        histo_value = [0]
        final_histo = []
        
        specHistFeature = data[data.Feature == f]
        orderData = specHistFeature.sort_values(by = ['Split point'], ignore_index = True)
        for i, s in enumerate(orderData['Split point']):
            if s not in split_points:
                split_points.append(s)
                right_side = [histo_value[-1] + float(orderData.loc[i, 'Right leaf value'])]
                left_side = [h + float(orderData.loc[i, 'Left leaf value']) for h in histo_value]
                histo_value = left_side + right_side
            else:
                left_side = [h + float(orderData.loc[i, 'Left leaf value']) for h in histo_value[:-1]]
                right_side = [histo_value[-1] + float(orderData.loc[i, 'Right leaf value'])]
                histo_value = left_side + right_side
                
        histoDataForPlot[f] = {'Splitting points': split_points,
                               'Histogram values': histo_value}
        
    return histoDataForPlot

def buildHistoLine(split_value_histo, x_min, x_max, num_points):
    
    x_values = np.linspace(x_min, x_max, num_points)
    histo_line = []
    i = 0
    max_i = len(split_value_histo['Splitting points'])
    for x in x_values:
        if x < float(split_value_histo['Splitting points'][i]):
            histo_line += [float(split_value_histo['Histogram values'][i])]
        else:
            histo_line += [float(split_value_histo['Histogram values'][i+1])]
            if i < max_i-1:
                i+=1
    
    return x_values, histo_line  

def plotHisto(model, X, units, Betas = None , withPointDist = False, model_unconstrained = None):
    
    params = json.loads(model.save_config())['learner']['gradient_booster']['updater']['grow_colmaker']['train_param']
    
    if params['eta'] is not None:
        lr = float(params['eta'])
    else:
        lr = 0.3
    
    if params['lambda'] is None:
        raise Exception('L1 and L2 regularization are not supported, please set alpha and lambda to 0 in the classifier')
    elif params['alpha'] is None:
        raise Exception('L1 and L2 regularization are not supported, please set alpha and lambda to 0 in the classifier')
    elif (float(params['alpha']) + float(params['lambda'])) != 0:
        raise Exception('L1 and L2 regularization are not supported, please set alpha and lambda to 0 in the classifier')
        
    params_unc = json.loads(model_unconstrained.save_config())['learner']['gradient_booster']['updater']['grow_colmaker']['train_param']
    
    if params_unc['eta'] is not None:
        lr_unc = float(params_unc['eta'])
    else:
        lr_unc = 0.3
    
    if params_unc['lambda'] is None:
        raise Exception('L1 and L2 regularization are not supported, please set alpha and lambda to 0 in the classifier')
    elif params_unc['alpha'] is None:
        raise Exception('L1 and L2 regularization are not supported, please set alpha and lambda to 0 in the classifier')
    elif (float(params_unc['alpha']) + float(params_unc['lambda'])) != 0:
        raise Exception('L1 and L2 regularization are not supported, please set alpha and lambda to 0 in the classifier')
    
    histoData = getHistoData(model)
    histoData_df = pd.DataFrame(np.reshape(histoData, (-1, 4)), columns= ['Feature', 'Split point', 'Left leaf value', 'Right leaf value'])
    
    dict_histograms = arrangeHistoDataForPlot(histoData_df)
    
    if model_unconstrained is not None:
        histoData_unc = getHistoData(model_unconstrained)
        histoData_df_unc = pd.DataFrame(np.reshape(histoData_unc, (-1, 4)), columns= ['Feature', 'Split point', 'Left leaf value', 'Right leaf value'])
    
        dict_histograms_unc = arrangeHistoDataForPlot(histoData_df_unc)
    
    sns.set_theme()
    
    for i, f in enumerate(histoData_df.Feature.unique()):
        
        x, histo_line_lr = buildHistoLine(dict_histograms[f], 0, 1.1*max(X[f]), 1000)
        
        histo_line = [h/lr for h in histo_line_lr]
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=x, y=histo_line, lw=2)
        plt.title('Influence of {} on the predictive function (utility)'.format(f), fontdict={'fontsize':  16})
        plt.xlabel('{} [{}]'.format(f, units[i]))
        plt.ylabel('Utility')          

        
        if model_unconstrained is not None:
            _, histo_line_unc_lr = buildHistoLine(dict_histograms_unc[f], 0, 1.1*max(X[f]), 1000)
            histo_line_unc =  [h_unc/lr_unc for h_unc in histo_line_unc_lr]
            sns.lineplot(x=x, y=histo_line_unc, lw=2)      
        
        if Betas is not None:
            sns.lineplot(x=x, y=Betas[i]*x)
            
        if withPointDist:
            sns.scatterplot(x=x, y=0*x, s=100, alpha=0.1)
        
        if Betas is not None:
            if model_unconstrained is not None:
                if withPointDist:
                    plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained', 'With RUM', 'Data'])
                else:
                    plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained', 'With RUM'])
            else:
                if withPointDist:
                    plt.legend(labels = ['With GBM constrained', 'With RUM', 'Data'])
                else:
                    plt.legend(labels = ['With GBM constrained', 'With RUM'])
        else:
            if model_unconstrained is not None:
                if withPointDist:
                    plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained', 'Data'])
                else:
                    plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained'])
            else:
                if withPointDist:
                    plt.legend(labels = ['With GBM constrained', 'Data'])
                else:
                    plt.legend(labels = ['With GBM constrained'])
                    
        plt.show()

