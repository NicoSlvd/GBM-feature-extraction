# coding: utf-8
"""Library with training routines of LightGBM."""
import collections
import copy
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.special import softmax

from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from lightgbm import callback
from lightgbm.basic import Booster, Dataset, LightGBMError, _ConfigAliases, _InnerPredictor, _choose_param_value, _log_warning
from lightgbm.compat import SKLEARN_INSTALLED, _LGBMGroupKFold, _LGBMStratifiedKFold

_LGBM_CustomObjectiveFunction = Callable[
    [Union[List, np.ndarray], Dataset],
    Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]
]
_LGBM_CustomMetricFunction = Callable[
    [Union[List, np.ndarray], Dataset],
    Tuple[str, float, bool]
]




class RUMBooster:
    """RUMBooster for doing Random Utility Modelling in LightGBM.
    
    Auxiliary data structure to implement boosters of ``rum_train()`` function for multiclass classification.
    This class has the same methods as Booster class.
    All method calls, except for the following methods, are actually performed for underlying Boosters.

    - ``model_from_string()``
    - ``model_to_string()``
    - ``save_model()``

    Attributes
    ----------
    boosters : list of Booster
        The list of underlying fitted models.
    valid_sets : None
        Validation sets of the RUMBooster. By default None, to avoid computing cross entropy if there are no 
        validation sets.
    """
    def __init__(self, model_file = None):
        """Initialize the RUMBooster.

        Parameters
        ----------
        model_file : str, pathlib.Path or None, optional (default=None)
            Path to the RUMBooster model file.
        """
        self.boosters = []
        self.valid_sets = None

        if model_file is not None:
            with open(model_file, "r") as file:
                self._from_dict(json.load(file))

    
    def f_obj(
            self,
            _,
            train_set
        ):
            """
            Objective function of the binary classification boosters, but based on softmax predictions

            Parameters
            ----------
            train_set: Dataset
                Training set used to train the jth booster. It means that it is not the full training set but rather
                another dataset containing the relevant features for that utility. It is the jth dataset in the
                RUMBooster object.

            """
            j = self._current_j
            preds = self._preds[:,j]
            factor = self.num_classes/(self.num_classes-1)
            eps = 1e-6
            labels = train_set.get_label()
            grad = preds - labels
            hess = np.maximum(factor * preds * (1 - preds), eps)
            return grad, hess
            
    def predict(
        self,
        data,
        start_iteration: int = 0,
        num_iteration: int = -1,
        raw_score: bool = True,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        utilities: bool = False,
        piece_wise: bool = False
    ):
        """Predict logic.

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, H2O DataTable's Frame or scipy.sparse
            Data source for prediction.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM).
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
        num_iteration : int, optional (default=-1)
            Iteration used for prediction.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.
        data_has_header : bool, optional (default=False)
            Whether data has header.
            Used only for txt data.
        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        utilities : bool, optional (default=True)
            If True, return raw utilities for each class, without generating probabilities.
        piece_wise: bool, optional (default=False)
            If True, use piece-wise utility instead of stairs utility.

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        U = []
        
        #compute utilities with corresponding features
        if not piece_wise:
            #separate features in J corresponding datasets
            new_data, _ = self._preprocess_data(data, return_data=True)
            for k, booster in enumerate(self.boosters):
                U.append(booster.predict(new_data[k].get_data(), 
                                start_iteration, 
                                num_iteration, 
                                raw_score, 
                                pred_leaf, 
                                pred_contrib,
                                data_has_header,
                                validate_features))
        else:
            U = self.pw_predict(data.get_data(), utility= True)

        preds = np.array(U).T

        #softmax
        if not utilities:
            preds = softmax(preds, axis=1)
   
        return preds
    
    def _inner_predict(
        self,
        data = None,
        start_iteration: int = 0,
        num_iteration: int = -1,
        raw_score: bool = True,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        utilities: bool = False,
        piece_wise: bool = False
    ):
        """Inner predict logic, the dataset is not assumed to be already build. Should not be used in public

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, H2O DataTable's Frame or scipy.sparse
            Data source for prediction.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM).
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
        num_iteration : int, optional (default=-1)
            Iteration used for prediction.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.
        data_has_header : bool, optional (default=False)
            Whether data has header.
            Used only for txt data.
        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        utilities : bool, optional (default=True)
            If True, return raw utilities for each class, without generating probabilities. 

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        U = []

        #getting dataset
        if data is None:
            data = self.train_set

        #compute utilities with corresponding features
        if not piece_wise:
            for k, booster in enumerate(self.boosters):
                U.append(booster.predict(data[k].get_data(), 
                                start_iteration, 
                                num_iteration, 
                                raw_score, 
                                pred_leaf, 
                                pred_contrib,
                                data_has_header,
                                validate_features))
        else:
            U = self.pw_predict(data, utility= True)

        preds = np.array(U).T

        #softmax
        if not utilities:
            preds = softmax(preds, axis=1)

        return preds
    
    def _stablesoftmax(self, x):
        """Compute the softmax of vector x in a numerically stable way."""
        shiftx = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shiftx)
        return exps / exps.sum(axis=1)[:,None]
    
    def _get_mid_pos(self, train_data, split_points):
        '''
        return midpoint in-between two split points for a specific feature
        '''
        #getting position in the middle of splitting points intervals
        if len(split_points) > 1:
            mid_pos = [(sp2 + sp1)/2 for sp2, sp1 in zip(split_points[:-1], split_points[1:])]
        else:
            mid_pos = []
        
        mid_pos.insert(0, min(train_data)) #adding first point
        mid_pos.append(max(train_data)) #adding last point

        return mid_pos
    
    def _get_slope(self, mid_pos, leaf_values):
        '''
        get slope of the piece-wise utility function for a specific feature
        '''
        if len(leaf_values) <= 1:
            return 0
        
        slope = [(leaf_values[i+1]-leaf_values[i])/(mid_pos[i+1]-mid_pos[i]) for i in range(0, len(mid_pos)-1)]
        slope.insert(0, 0) #adding first slope
        slope.append(0) #adding last slope

        return slope

    def _stairs_to_pw(self, train_data, data_to_transform = None, util_for_plot = False):
        '''
        Transform a stair output to a piecewise linear prediction
        '''
        if type(train_data) is list:
            new_train_data = train_data[0].get_data()
            for data in train_data[1:]:
                new_train_data = new_train_data.join(data.get_data(), lsuffix='DROP').filter(regex="^(?!.*DROP)")

            train_data = new_train_data

        if data_to_transform is None:
            data_to_transform = train_data
        weights = self.weights_to_plot()
        pw_utility = []
        for u in weights:
            if util_for_plot:
                pw_util = []
            else:
                pw_util = np.zeros(data_to_transform.iloc[:, 0].shape)

            for f in weights[u]:
                leaf_values = weights[u][f]['Histogram values']
                split_points =  weights[u][f]['Splitting points']
                transf_data_arr = np.array(data_to_transform[f])

                if len(split_points) < 1:
                    break
                
                if self.rum_structure[int(u)]['columns'].index(f) not in self.rum_structure[int(u)]['categorical_feature']:

                    mid_pos = self._get_mid_pos(self.train_set[int(u)].get_data()[f],split_points)
                    
                    slope = self._get_slope(mid_pos, leaf_values)

                    conds = [(mp1 <= transf_data_arr) & (transf_data_arr < mp2) for mp1, mp2 in zip(mid_pos[:-1], mid_pos[1:])]
                    conds.insert(0, transf_data_arr<mid_pos[0])
                    conds.append(transf_data_arr >= mid_pos[-1])

                    values = [lambda x, j=j: leaf_values[j] + slope[j+1] * (x - mid_pos[j]) for j in range(0, len(leaf_values))]
                    values.insert(0, leaf_values[0])
                else:
                    conds = [(sp1 <= transf_data_arr) & (transf_data_arr < sp2) for sp1, sp2 in zip(split_points[:-1], split_points[1:])]
                    conds.insert(0, transf_data_arr < split_points[0])
                    conds.append(transf_data_arr >= split_points[-1])
                    values = leaf_values
                
                if util_for_plot:
                    pw_util.append(np.piecewise(transf_data_arr, conds, values))
                else:
                    pw_util = np.add(pw_util, np.piecewise(transf_data_arr, conds, values))
            pw_utility.append(pw_util)

        return pw_utility

    def pw_predict(self, data, data_to_transform = None, utility = False):
        '''
        compute predictions with piece-wise utility
        '''
        U = self._stairs_to_pw(data, data_to_transform)

        if utility:
            return U
        
        return softmax(np.array(U).T, axis=1)


    def accuracy(self, preds, labels):
        """
        compute accuracy of the model
        """
        labels_pred = np.argmax(preds, axis=1)
        return np.mean(labels_pred == labels)
    
    def cross_entropy(self, preds, labels):
        """
        Compute cross entropy of the RUMBooster model for the given predictions and data
        
        Parameters
        ----------
        preds: ndarray
            Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
            to the prediction of data point i to belong to class j
        labels: ndarray
            The labels of the original dataset, as int
        Returns
        -------
        Cross entropy : float
        """
        c_entr = 0
        for i, l in enumerate(labels):
            c_entr += np.log(preds[i, l])

        return - c_entr / len(labels)
    
    def _preprocess_data(self, data, reduced_valid_set = None, return_data = False):
        """Set up J training (and, if specified, validation) datasets"""
        train_set_J = []
        reduced_valid_sets_J = []

        #to access data
        data.construct()
        for j, struct in enumerate(self.rum_structure):
            if struct:
                if 'columns' in struct:
                    train_set_j_data = data.get_data()[struct['columns']] #only relevant features for the j booster
                    new_label = np.array([1 if l == j else 0 for l in data.get_label()]) #new binary label
                    if self.with_linear_trees:
                        train_set_j = Dataset(train_set_j_data, label=new_label, free_raw_data=False, params={'linear_trees':True})
                    else:
                        train_set_j = Dataset(train_set_j_data, label=new_label, free_raw_data=False)
                    train_set_j.construct()
                    if reduced_valid_set is not None:
                        reduced_valid_sets_j = []
                        for valid_set in reduced_valid_set:
                            valid_set.construct()
                            valid_set_j_data = valid_set.get_data()[struct['columns']] #only relevant features for the j booster
                            label_valid = valid_set.get_label()#new binary label
                            valid_set_j = Dataset(valid_set_j_data, label=label_valid, reference= train_set_j, free_raw_data=False)
                            valid_set_j.construct()
                            reduced_valid_sets_j.append(valid_set_j)

                else:
                    train_set_j = data
                    if reduced_valid_set is not None:
                        for valid_set in reduced_valid_set:
                            reduced_valid_sets_j.append(valid_set)

            train_set_J.append(train_set_j)
            if reduced_valid_set is not None:
                reduced_valid_sets_J.append(reduced_valid_sets_j)

        #storing them in the RUMBooster object
        self.train_set = train_set_J
        self.valid_sets = np.array(reduced_valid_sets_J).T.tolist()
        if return_data:
            return train_set_J, reduced_valid_sets_J
    
    def _preprocess_params(self, params, return_params=False):
        """Set up J set of parameters"""
        params_J = []
        self.num_classes = params['num_classes']
        for struct in self.rum_structure:
            params_j = copy.deepcopy(params)
            params_j['objective'] = 'binary'
            params_j['num_classes'] = 1
            if 'linear_tree' in params_j:
                self.with_linear_trees = params_j.pop('linear_tree')
            else:
                self.with_linear_trees = False
            if struct:
                if len(struct['monotone_constraints'])>0:
                    params_j['monotone_constraints'] = struct['monotone_constraints']
                if len(struct['interaction_constraints'])>0:
                    params_j['interaction_constraints'] = struct['interaction_constraints']
                if 'categorical_feature' in struct:
                    params_j['categorical_feature'] = struct['categorical_feature']

            params_J.append(params_j)

        self.params = params_J
        if return_params:
            return params_J
        
    def _preprocess_valids(self, train_set, params, valid_sets = None, valid_names = None):
        """Set up validation sets"""
        #construct training set to access data
        train_set.construct()

        #initializing variables
        is_valid_contain_train = False
        train_data_name = "training"
        reduced_valid_sets = []
        name_valid_sets = []

        if valid_sets is not None:
            if isinstance(valid_sets, Dataset):
                valid_sets = [valid_sets]
            if isinstance(valid_names, str):
                valid_names = [valid_names]
            for i, valid_data in enumerate(valid_sets):
                # reduce cost for prediction training data
                if valid_data is train_set:
                    is_valid_contain_train = True
                    if valid_names is not None:
                        train_data_name = valid_names[i]
                    continue
                if not isinstance(valid_data, Dataset):
                    raise TypeError("Training only accepts Dataset object")
                reduced_valid_sets.append(valid_data._update_params(params).set_reference(train_set))
                if valid_names is not None and len(valid_names) > i:
                    name_valid_sets.append(valid_names[i])
                else:
                    name_valid_sets.append(f'valid_{i}')

        return reduced_valid_sets, name_valid_sets, is_valid_contain_train, train_data_name
        
    
    def _construct_boosters(self, train_data_name = "Training", is_valid_contain_train = False,
                            name_valid_sets = None):
        """Construct boosters of the RUMBooster model with corresponding set of parameters and training features"""
        #getting parameters, training, and validation sets
        params_J = self.params
        train_set_J = self.train_set
        reduced_valid_sets_J = self.valid_sets

        for j in range(len(self.rum_structure)):
            try: 
                #construct binary booster
                booster = Booster(params=params_J[j], train_set=train_set_J[j])
                if is_valid_contain_train:
                    booster.set_train_data_name(train_data_name)
                for valid_set, name_valid_set in zip(reduced_valid_sets_J, name_valid_sets):
                    booster.add_valid(valid_set[j], name_valid_set)
            finally:
                train_set_J[j]._reverse_update_params()
                for valid_set in reduced_valid_sets_J:
                    valid_set[j]._reverse_update_params()
            booster.best_iteration = 0
            self._append(booster)
        self.best_iteration = 0
        self.best_score = 1000000


    def _append(self, booster: Booster) -> None:
        """Add a booster to RUMBooster."""
        self.boosters.append(booster)

    def _from_dict(self, models: Dict[str, Any]) -> None:
        """Load RUMBooster from dict."""
        self.best_iteration = models["best_iteration"]
        self.best_score = models["best_score"]
        self.boosters = []
        for model_str in models["boosters"]:
            self._append(Booster(model_str=model_str))

    def _to_dict(self, num_iteration: Optional[int], start_iteration: int, importance_type: str) -> Dict[str, Any]:
        """Serialize RUMBooster to dict."""
        models_str = []
        for booster in self.boosters:
            models_str.append(booster.model_to_string(num_iteration=num_iteration, start_iteration=start_iteration,
                                                      importance_type=importance_type))
        return {"boosters": models_str, "best_iteration": self.best_iteration, "best_score": self.best_score}

    def __getattr__(self, name: str) -> Callable[[Any, Any], List[Any]]:
        """Redirect methods call of RUMBooster."""
        def handler_function(*args: Any, **kwargs: Any) -> List[Any]:
            """Call methods with each booster, and concatenate their results."""
            ret = []
            for booster in self.boosters:
                ret.append(getattr(booster, name)(*args, **kwargs))
            return ret
        return handler_function

    def __getstate__(self) -> Dict[str, Any]:
        return vars(self)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)

    def getweights(self, model = None):
        """
        get leaf values from a RUMBooster or LightGBM model

        Parameters
        ----------
        model: lightGBM model

        Returns
        -------
        weights_df: DataFrame
            DataFrame containing all split points and their corresponding left and right leaves value, 
            for all features
        """
        #using self object or a given model
        if model is None:
            model_json = self.dump_model()
        else:
            model_json = [model.dump_model()]

        weights = []

        for i, b in enumerate(model_json):
            feature_names = b['feature_names']
            for trees in b['tree_info']:
                feature = feature_names[trees['tree_structure']['split_feature']]
                split_point = trees['tree_structure']['threshold']
                left_leaf_value = trees['tree_structure']['left_child']['leaf_value']
                right_leaf_value = trees['tree_structure']['right_child']['leaf_value']
                weights.append([feature, split_point, left_leaf_value, right_leaf_value, i])

        weights_df = pd.DataFrame(weights, columns= ['Feature', 'Split point', 'Left leaf value', 'Right leaf value', 'Utility'])
        return weights_df
    
    def _create_name(self, features):
        """Create new feature names from a list of feature names"""
        new_name = features[0]
        for f_name in features[1:]:
            new_name += '-'+f_name
        return new_name

    
    def _get_child(self, weights, weights_2d, tree, split_points, features, feature_names, i, direction = None):
        """Dig into the tree to get splitting points, features, left and right leaves values"""
        min_r = 0
        max_r = 10000
        if feature_names[tree['split_feature']] not in features:
            features.append(feature_names[tree['split_feature']])
        split_points.append(tree['threshold'])
        if 'leaf_value' in tree['left_child'] and 'leaf_value' in tree['right_child']:
            if direction is None:
                weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
            elif direction == 'left':
                if len(features) == 1:
                    weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
                    weights.append([feature_names[tree['split_feature']], split_points[0], 0, -tree['right_child']['leaf_value'], i])
                else:
                    feature_name = self._create_name(features)
                    weights_2d.append([feature_name, (min_r, split_points[0]), (min_r, tree['threshold']), tree['left_child']['leaf_value'], i])
                    weights_2d.append([feature_name, (min_r, split_points[0]), (tree['threshold'], max_r), tree['right_child']['leaf_value'], i])
                    if len(features) > 1:
                        features.pop(-1)
                        split_points.pop(-1)
            elif direction == 'right':
                if len(features) == 1:
                    weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
                    weights.append([feature_names[tree['split_feature']], split_points[0], -tree['left_child']['leaf_value'], 0, i])
                else:
                    feature_name = self._create_name(features)
                    weights_2d.append([feature_name, (split_points[0], max_r), (min_r, tree['threshold']), tree['left_child']['leaf_value'], i])
                    weights_2d.append([feature_name, (split_points[0], max_r), (tree['threshold'], max_r), tree['right_child']['leaf_value'], i])
        elif 'leaf_value' in tree['left_child']:
            weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], 0, i])
            self._get_child(weights, weights_2d, tree['right_child'], split_points, features, feature_names, i, direction='right')
        elif 'leaf_value' in tree['right_child']:
            weights.append([feature_names[tree['split_feature']], tree['threshold'], 0, tree['right_child']['leaf_value'], i])
            self._get_child(weights, weights_2d, tree['left_child'], split_points, features, feature_names, i, direction='left')
        else:
            self._get_child(weights, weights_2d, tree['left_child'], split_points, features, feature_names, i, direction='left')
            self._get_child(weights, weights_2d, tree['right_child'], split_points, features, feature_names, i, direction='right')
        
        

    def getweights_v2(self, model = None):
        """
        get leaf values from a RUMBooster or LightGBM model

        Parameters
        ----------
        model: lightGBM model

        Returns
        -------
        weights_df: DataFrame
            DataFrame containing all split points and their corresponding left and right leaves value, 
            for all features
        """
        #using self object or a given model
        if model is None:
            model_json = self.dump_model()
        else:
            model_json = [model.dump_model()]

        weights = []
        weights_2d = []

        for i, b in enumerate(model_json):
            feature_names = b['feature_names']
            for trees in b['tree_info']:
                features = []
                split_points = []

                self._get_child(weights, weights_2d, trees['tree_structure'], split_points, features, feature_names, i)

        weights_df = pd.DataFrame(weights, columns= ['Feature', 'Split point', 'Left leaf value', 'Right leaf value', 'Utility'])
        weights_2d_df = pd.DataFrame(weights_2d, columns=['Feature', 'higher_lvl_range', 'lower_lvl_range', 'area_value', 'Utility'])
        return weights_df, weights_2d_df
    
    def weights_to_plot_v2(self, model = None):
        """
        Arrange weights by ascending splitting points and cumulative sum of weights

        Parameters
        ----------
        model: lightGBM model

        Returns
        -------
        weights_for_plot: dict
            Dictionary containing splitting points and corresponding cumulative weights value for all features
        """

        #get raw weights
        if model is None:
            weights, _ = self.getweights_v2()
        else:
            weights, _ = self.getweights_v2(model=model)

        weights_for_plot = {}
        #for all features
        for i in weights.Utility.unique():
            weights_for_plot[str(i)] = {}
            
            for f in weights[weights.Utility == i].Feature.unique():
                
                split_points = []
                function_value = [0]

                #getting values related to the corresponding utility
                weights_util = weights[weights.Utility == i]
                
                #sort by ascending order
                feature_data = weights_util[weights_util.Feature == f]
                ordered_data = feature_data.sort_values(by = ['Split point'], ignore_index = True)
                for j, s in enumerate(ordered_data['Split point']):
                    #new split point
                    if s not in split_points:
                        split_points.append(s)
                        #add a new right leaf value to the current right side value
                        function_value.append(function_value[-1] + float(ordered_data.loc[j, 'Right leaf value']))
                        #add left leaf value to all other current left leaf values
                        function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
                    else:
                        #add right leaf value to the current right side value
                        function_value[-1] += float(ordered_data.loc[j, 'Right leaf value'])
                        #add left leaf value to all other current left leaf values
                        function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
                        
                weights_for_plot[str(i)][f] = {'Splitting points': split_points,
                                            'Histogram values': function_value}
                    
        return weights_for_plot

    def weights_to_plot(self, model = None):
        """
        Arrange weights by ascending splitting points and cumulative sum of weights

        Parameters
        ----------
        model: lightGBM model

        Returns
        -------
        weights_for_plot: dict
            Dictionary containing splitting points and corresponding cumulative weights value for all features
        """

        #get raw weights
        if model is None:
            weights = self.getweights()
        else:
            weights = self.getweights(model=model)

        weights_for_plot = {}
        weights_for_plot_double = {}
        #for all features
        for i in weights.Utility.unique():
            weights_for_plot[str(i)] = {}
            weights_for_plot_double[str(i)] = {}
            
            for f in weights[weights.Utility == i].Feature.unique():
                
                split_points = []
                function_value = [0]

                #getting values related to the corresponding utility
                weights_util = weights[weights.Utility == i]
                
                #sort by ascending order
                feature_data = weights_util[weights_util.Feature == f]
                ordered_data = feature_data.sort_values(by = ['Split point'], ignore_index = True)
                for j, s in enumerate(ordered_data['Split point']):
                    #new split point
                    if s not in split_points:
                        split_points.append(s)
                        #add a new right leaf value to the current right side value
                        function_value.append(function_value[-1] + float(ordered_data.loc[j, 'Right leaf value']))
                        #add left leaf value to all other current left leaf values
                        function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
                    else:
                        #add right leaf value to the current right side value
                        function_value[-1] += float(ordered_data.loc[j, 'Right leaf value'])
                        #add left leaf value to all other current left leaf values
                        function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
                        
                weights_for_plot[str(i)][f] = {'Splitting points': split_points,
                                            'Histogram values': function_value}
                    
        return weights_for_plot
    
    def non_lin_function(self, weights_ordered, x_min, x_max, num_points):
        """
        Create the nonlinear function for parameters, from weights ordered by ascending splitting points

        Parameters
        ----------
        weights_ordered : dict
            Dictionary containing splitting points and corresponding cumulative weights value for a specific 
            feature's parameter
        x_min : float, int
            Minimum x value for which the nonlinear function is computed
        x_max : float, int
            Maximum x value for which the nonlinear function is computed
        num_points: int
            Number of points used to draw the nonlinear function line

        Returns
        -------
        x_values: list
            X values for which the function will be plotted
        nonlin_function: list
            Values of the function at the corresponding x points
        """
        #create x points
        x_values = np.linspace(x_min, x_max, num_points)
        nonlin_function = []
        i = 0
        max_i = len(weights_ordered['Splitting points']) #all splitting points

        #handling no split points
        if max_i == 0:
            return x_values, float(weights_ordered['Histogram values'][i])*x_values

        for x in x_values:
            #compute the value of the function at x according to the weights value in between splitting points
            if x < float(weights_ordered['Splitting points'][i]):
                nonlin_function += [float(weights_ordered['Histogram values'][i])]
            else:
                nonlin_function += [float(weights_ordered['Histogram values'][i+1])]
                #go to next splitting points
                if i < max_i-1:
                    i+=1
        
        return x_values, nonlin_function
    
    def function_2d(self, weights_2d, x_vect, y_vect):
        """
        Create the nonlinear contour plot for parameters, from weights gathered in getweights_v2

        Parameters
        ----------
        weights_2d : dict
            Pandas DataFrame containing all possible rectangles with their corresponding area values, for the given feature and utility
        x_vect : np.linspace
            Vector of higher level feature
        y_vect : np.linspace
            Vector of lower level feature

        Returns
        -------
        contour_plot_values: np.darray
            Array with values at (x,y) points
        """
        contour_plot_values = np.zeros(shape=(len(x_vect), len(y_vect)))

        for k in range(len(weights_2d.index)):
            y_check = True
            for i in range(len(x_vect)):
                if (x_vect[i] >= weights_2d['higher_lvl_range'].iloc[k][1]) | (not y_check):
                    break

                if x_vect[i] >= weights_2d['higher_lvl_range'].iloc[k][0]:
                    for j in range(len(y_vect)):
                        if y_vect[j] >= weights_2d['lower_lvl_range'].iloc[k][1]:
                            y_check = False
                            break
                        if y_vect[j] >= weights_2d['lower_lvl_range'].iloc[k][0]:
                            if (weights_2d['lower_lvl_range'].iloc[k][1] == 10000) and (weights_2d['higher_lvl_range'].iloc[k][1] == 10000):
                                contour_plot_values[i:, j:] += weights_2d['area_value'].iloc[k]
                                y_check = False
                                break
                            else:
                                contour_plot_values[i, j] += weights_2d['area_value'].iloc[k]

        return contour_plot_values

    def plot_2d(self, feature1, feature2, max1, max2, model = None):

        _, weights_2d = self.getweights_v2(model = model)

        name1 = feature1 + "-" + feature2
        name2 = feature2 + "-" + feature1

        x_vect = np.linspace(0, max1, 75)
        y_vect = np.linspace(0, max2, 75)
        
        contour_plot1 = self.function_2d(weights_2d[weights_2d.Feature==name1], x_vect, y_vect)
        contour_plot2 = self.function_2d(weights_2d[weights_2d.Feature==name2], y_vect, x_vect)

        contour_plot = contour_plot1 + contour_plot2.T

        X, Y = np.meshgrid(x_vect, y_vect)

        fig1, ax1 = plt.subplots(layout='constrained')

        sns.set_theme()
        c_plot = ax1.contourf(X, Y, contour_plot, linewidths=0)

        ax1.set_title('Impact of {} and {} on the utility function'.format(feature1, feature2))
        ax1.set_xlabel('{}'.format(feature1))
        ax1.set_ylabel('{}'.format(feature2))

        cbar = fig1.colorbar(c_plot)
        cbar.ax.set_ylabel('Utility value')

        plt.show()
    
    def plot_parameters(self, params, X, utility_names, Betas = None , withPointDist = False, model_unconstrained = None, 
                        params_unc = None, with_pw = False, with_stairs = True, save_figure=False):
        """
        Plot the non linear impact of parameters on the utility function. When specified, unconstrained parameters
        and parameters from a RUM model can be added to the plot.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters used to train the RUM booster.
        X : pandas dataframe
            Features used to train the model, in a pandas dataframe.
        utility_name : dict
            Dictionary mapping utilities to their names.
        Betas : list, optional (default = None)
            List of beta parameters value from a RUM. They should be listed in the same order as 
            in the RUMBooster model.
        withPointDist: Bool, optional (default = False)
            If True, the distribution of the training samples for the corresponding features will be plot 
            on the x axis
        model_unconstrained: LightGBM model, optional (default = None)
            The unconstrained model. Must be trained and compatible with dump_model().
        params_unc: dict, optional (default = None)
            Dictionary containing parameters used to train the unconstrained model
        with_pw: bool, optional (default = False)
            If the piece-wise function should be included in the graph
        with_stairs: bool, optional (default = True)
            If False, continuous feature graphs are not drawn with stairs
        """
        #getting learning rate
        # if params['learning_rate'] is not None:
        #     lr = float(params['learning_rate'])
        # else:
        #     lr = 0.3
        
        # if model_unconstrained is not None:
        #     if params_unc['learning_rate'] is not None:
        #         lr_unc = float(params_unc['learning_rate'])
        #     else:
        #         lr_unc = 0.3
        
        #get and prepare weights
        weights_arranged = self.weights_to_plot_v2()

        if with_pw:
            pw_func = self.plot_util_pw(X)
        
        if model_unconstrained is not None:
            weights_arranged_unc = self.weights_to_plot_v2(model=model_unconstrained)

        sns.set_theme()
        
        #for all features parameters
        for u in weights_arranged:
            for i, f in enumerate(weights_arranged[u]):
                
                #create nonlinear plot
                x, non_lin_func = self.non_lin_function(weights_arranged[u][f], 0, 1.05*max(X[f]), 10000)
                
                #non_lin_func_with_lr = [h/lr for h in non_lin_func]
                
                #plot parameters
                plt.figure(figsize=(10, 6))

                                    
                plt.title('Influence of {} on the predictive function ({} utility)'.format(f, utility_names[u]), fontdict={'fontsize':  16})
                if 'dur' in f:
                    plt.xlabel('{} [h]'.format(f))
                elif 'time' in f:
                    plt.xlabel('{} [h]'.format(f))
                elif 'cost' in f:
                    plt.xlabel('{} [£]'.format(f))
                elif 'distance' in f:
                    plt.xlabel('{} [m]'.format(f))
                else:
                    plt.xlabel('{}'.format(f))

                plt.ylabel('{} utility'.format(utility_names[u]))

                if with_stairs | (self.rum_structure[int(u)]['columns'].index(f) in self.rum_structure[int(u)]['categorical_feature']):
                    sns.lineplot(x=x, y=non_lin_func, lw=2)

                #plot unconstrained model parameters
                if model_unconstrained is not None:
                    _, non_lin_func_unc = self.non_lin_function(weights_arranged_unc[u][f], 0, 1.05*max(X[f]), 10000)
                    #non_lin_func_with_lr_unc =  [h_unc/lr_unc for h_unc in non_lin_func_unc]
                    sns.lineplot(x=x, y=non_lin_func_unc, lw=2)

                if (with_pw) & (self.rum_structure[int(u)]['columns'].index(f)  not in self.rum_structure[int(u)]['categorical_feature']):
                    #pw_func_with_lr = [pw/lr for pw in pw_func[int(u)][i]]
                    sns.lineplot(x=x, y=pw_func[int(u)][i], lw=2)
                
                #plot RUM parameters
                if Betas is not None:
                    sns.lineplot(x=x, y=Betas[i]*x)
                
                #plot data distribution
                if withPointDist:
                    sns.scatterplot(x=x, y=0*x, s=100, alpha=0.1)
                
                #legend
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
                        elif with_pw:
                            plt.legend(labels = ['With GBM constrained', 'With piece-wise linear function'])
                        else:
                            plt.legend(labels = ['RUMBooster'])

                if save_figure:
                    plt.savefig('Figures/rumbooster_v3_lr3e-1 {} utility, {} feature.png'.format(utility_names[u], f))

                plt.show()

    def plot_util(self, data_train, points=10000):
        sns.set_theme()
        for j, struct in enumerate(self.rum_structure):
            booster = self.boosters[j]
            for i, f in enumerate(struct['columns']):
                xin = np.zeros(shape = (points, len(struct['columns'])))
                xin[:, i] = np.linspace(0,1.05*max(data_train[f]),points)
                
                ypred = booster.predict(xin)
                plt.figure()
                plt.plot(np.linspace(0,1.05*max(data_train[f]),points), ypred)
                plt.title(f)

    def plot_util_pw(self, data_train, points = 10000):
        '''
        plot the piece-wise utility function
        '''
        features = data_train.columns
        data_to_transform = {}
        for f in features:
            xi = np.linspace(0, 1.05*max(data_train[f]), points)
            data_to_transform[f] = xi

        data_to_transform = pd.DataFrame(data_to_transform)

        pw_func = self._stairs_to_pw(data_train, data_to_transform, util_for_plot = True)

        return pw_func
        

    def model_from_string(self, model_str: str):
        """Load RUMBooster from a string.

        Parameters
        ----------
        model_str : str
            Model will be loaded from this string.

        Returns
        -------
        self : RUMBooster
            Loaded RUMBooster object.
        """
        self._from_dict(json.loads(model_str))
        return self

    def model_to_string(
        self,
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = 'split'
    ) -> str:
        """Save RUMBooster to JSON string.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        str_repr : str
            JSON string representation of RUMBooster.
        """
        return json.dumps(self._to_dict(num_iteration, start_iteration, importance_type))

    def save_model(
        self,
        filename: Union[str, Path],
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = 'split'
    ) -> "RUMBooster":
        """Save RUMBooster to a file as JSON text.

        Parameters
        ----------
        filename : str or pathlib.Path
            Filename to save RUMBooster.
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        self : RUMBooster
            Returns self.
        """
        with open(filename, "w") as file:
            json.dump(self._to_dict(num_iteration, start_iteration, importance_type), file)

        return self

def rum_train(
    params: Dict[str, Any],
    train_set: Dataset,
    rum_structure: List[Dict[str, Any]],
    num_boost_round: int = 100,
    valid_sets: Optional[List[Dataset]] = None,
    valid_names: Optional[List[str]] = None,
    feval: Optional[Union[_LGBM_CustomMetricFunction, List[_LGBM_CustomMetricFunction]]] = None,
    init_model: Optional[Union[str, Path, Booster]] = None,
    feature_name: Union[List[str], str] = 'auto',
    categorical_feature: Union[List[str], List[int], str] = 'auto',
    keep_training_booster: bool = False,
    callbacks: Optional[List[Callable]] = None,
    pw_utility: bool = False
) -> RUMBooster:
    """Perform the RUM training with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for training. Values passed through ``params`` take precedence over those
        supplied via arguments.
    train_set : Dataset
        Data to be trained on.
    rum_structure : dict
        List of dictionaries specifying the RUM structure. 
        The list must contain one dictionary for each class, which describes the 
        utility structure for that class. 
        Each dictionary has three allowed keys. 
        'cols': list of columns included in that class
        'monotone_constraints': list of monotonic constraints on parameters
        'interaction_constraints': list of interaction constraints on features
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    valid_sets : list of Dataset, or None, optional (default=None)
        List of data to be evaluated on during training.
    valid_names : list of str, or None, optional (default=None)
        Names of ``valid_sets``.
    feval : callable, list of callable, or None, optional (default=None)
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, eval_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                The predicted values.
                For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes].
                If custom objective function is used, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            eval_data : Dataset
                A ``Dataset`` to evaluate.
            eval_name : str
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        To ignore the default metric corresponding to the used objective,
        set the ``metric`` parameter to the string ``"None"`` in ``params``.
    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.
    feature_name : list of str, or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of str or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
        Floating point numbers in categorical features will be rounded towards 0.
    keep_training_booster : bool, optional (default=False)
        Whether the returned Booster will be used to keep training.
        If False, the returned value will be converted into _InnerPredictor before returning.
        This means you won't be able to use ``eval``, ``eval_train`` or ``eval_valid`` methods of the returned Booster.
        When your model is very large and cause the memory error,
        you can try to set this param to ``True`` to avoid the model conversion performed during the internal call of ``model_to_string``.
        You can still use _InnerPredictor as ``init_model`` for future continue training.
    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    pw_utility: bool, optional (default=False)
        If true, compute continuous feature utility in a piece-wise linear way.

    Note
    ----
    A custom objective function can be provided for the ``objective`` parameter.
    It should accept two parameters: preds, train_data and return (grad, hess).

        preds : numpy 1-D array or numpy 2-D array (for multi-class task)
            The predicted values.
            Predicted values are returned before any transformation,
            e.g. they are raw margin instead of probability of positive class for binary task.
        train_data : Dataset
            The training dataset.
        grad : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of preds for each sample point.
        hess : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of preds for each sample point.

    For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes],
    and grad and hess should be returned in the same format.

    Returns
    -------
    rum_booster : RUMBooster
        The trained RUMBooster model.
    """
    # create predictor first
    params = copy.deepcopy(params)
    params = _choose_param_value(
        main_param_name='objective',
        params=params,
        default_value=None
    )
    fobj: Optional[_LGBM_CustomObjectiveFunction] = None
    if callable(params["objective"]):
        fobj = params["objective"]
        params["objective"] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            num_boost_round = params.pop(alias)
            _log_warning(f"Found `{alias}` in params. Will use it instead of argument")
    params["num_iterations"] = num_boost_round
    # setting early stopping via global params should be possible
    params = _choose_param_value(
        main_param_name="early_stopping_round",
        params=params,
        default_value=None
    )
    if params["early_stopping_round"] is None:
        params.pop("early_stopping_round")
    first_metric_only = params.get('first_metric_only', False)

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    predictor: Optional[_InnerPredictor] = None
    if isinstance(init_model, (str, Path)):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    init_iteration = predictor.num_total_iteration if predictor is not None else 0
    # check dataset
    if not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")

    train_set._update_params(params) \
             ._set_predictor(predictor) \
             .set_feature_name(feature_name) \
             .set_categorical_feature(categorical_feature)

    # process callbacks
    if callbacks is None:
        callbacks_set = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks_set = set(callbacks)

    #if "early_stopping_round" in params:
    #    callbacks_set.add(
    #        callback.early_stopping(
    #            stopping_rounds=params["early_stopping_round"],
    #            first_metric_only=first_metric_only,
    #            verbose=_choose_param_value(
    #                main_param_name="verbosity",
    #                params=params,
    #                default_value=1
    #            ).pop("verbosity") > 0
    #        )
    #    )

    callbacks_before_iter_set = {cb for cb in callbacks_set if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter_set = callbacks_set - callbacks_before_iter_set
    callbacks_before_iter = sorted(callbacks_before_iter_set, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter_set, key=attrgetter('order'))

    #construct boosters
    rum_booster = RUMBooster()
    reduced_valid_sets, \
    name_valid_sets, \
    is_valid_contain_train, \
    train_data_name = rum_booster._preprocess_valids(train_set, params, valid_sets) #prepare validation sets
    rum_booster.rum_structure = rum_structure #saving utility structure
    rum_booster._preprocess_params(params) #preparing J set of parameters
    rum_booster._preprocess_data(train_set, reduced_valid_sets, return_data=True) #preparing J datasets with relevant features
    rum_booster._construct_boosters(train_data_name, is_valid_contain_train, name_valid_sets) #building boosters with corresponding params and dataset

    #initial prediction for first iteration
    rum_booster._preds = rum_booster._inner_predict()

    #start training
    for i in range(init_iteration, init_iteration + num_boost_round):
        #initialising early stopping criterion
        early_stop_crit_all = [False] * params['num_classes']
        #updating all binary boosters of the rum_booster
        for j, booster in enumerate(rum_booster.boosters):
            for cb in callbacks_before_iter:
                cb(callback.CallbackEnv(model=booster,
                                        params=rum_booster.params[j],
                                        iteration=i,
                                        begin_iteration=init_iteration,
                                        end_iteration=init_iteration + num_boost_round,
                                        evaluation_result_list=None))       
    
            #update booster with custom binary objective function, and relevant features
            rum_booster._current_j = j
            booster.update(train_set=rum_booster.train_set[j], fobj=rum_booster.f_obj)
            
            # check evaluation result. (from lightGBM initial code, check on all J binary boosters)
            evaluation_result_list = []
            if valid_sets is not None:
                if is_valid_contain_train:
                    evaluation_result_list.extend(booster.eval_train(feval))
                evaluation_result_list.extend(booster.eval_valid(feval))
            try:
                for cb in callbacks_after_iter:
                    cb(callback.CallbackEnv(model=booster,
                                            params=rum_booster.params[j],
                                            iteration=i,
                                            begin_iteration=init_iteration,
                                            end_iteration=init_iteration + num_boost_round,
                                            evaluation_result_list=evaluation_result_list))
            except callback.EarlyStopException as earlyStopException:
                early_stop_crit_all[j] = True
                booster.best_iteration = earlyStopException.best_iteration + 1
                evaluation_result_list = earlyStopException.best_score

        #make predictions after boosting round to compute new cross entropy and for next iteration grad and hess
        #if i < 21:
        #    rum_booster._preds = rum_booster._inner_predict(piece_wise=False)
        #else:
        rum_booster._preds = rum_booster._inner_predict(piece_wise=pw_utility)
        #compute cross validation on training or validation test
        if valid_sets is not None:
            if is_valid_contain_train:
                cross_entropy = rum_booster.cross_entropy(rum_booster._preds, train_set.get_label().astype(int))
            else:
                for valid_set_J in rum_booster.valid_sets:
                    preds_valid = rum_booster._inner_predict(valid_set_J, piece_wise=pw_utility)
                    cross_entropy_train = rum_booster.cross_entropy(rum_booster._preds, train_set.get_label().astype(int))
                    cross_entropy = rum_booster.cross_entropy(preds_valid, valid_set_J[0].get_label().astype(int))
        
            if cross_entropy < rum_booster.best_score:
                rum_booster.best_score = cross_entropy
                rum_booster.best_iteration = i+1
        
            if (params['verbosity'] >= 1) and (i % 10 == 0):
                if is_valid_contain_train:
                    print('[{}] -- NCE value on train set: {}'.format(i + 1, cross_entropy))
                else:
                    print('[{}] -- NCE value on train set: {} \n     --  NCE value on test set: {}'.format(i + 1, cross_entropy_train, cross_entropy))
        
        #early stopping if early stopping criterion in all boosters
        if (params["early_stopping_round"] != 0) and (rum_booster.best_iteration + params["early_stopping_round"] < i + 1):
            print('Early stopping at iteration {}, with a best score of {}'.format(rum_booster.best_iteration, rum_booster.best_score))
            break

    for booster in rum_booster.boosters:
        booster.best_score = collections.defaultdict(collections.OrderedDict)
        for dataset_name, eval_name, score, _ in evaluation_result_list:
            booster.best_score[dataset_name][eval_name] = score
        if not keep_training_booster:
            booster.model_from_string(booster.model_to_string(), verbose='_silent_false').free_dataset()
    return rum_booster

class CVRUMBooster:
    """CVRUMBooster in LightGBM.

    Auxiliary data structure to hold and redirect all boosters of ``cv`` function.
    This class has the same methods as Booster class.
    All method calls are actually performed for underlying Boosters and then all returned results are returned in a list.

    Attributes
    ----------
    rum_boosters : list of RUMBooster
        The list of underlying fitted models.
    best_iteration : int
        The best iteration of fitted model.
    """

    def __init__(self):
        """Initialize the CVBooster.

        Generally, no need to instantiate manually.
        """
        self.rumboosters = []
        self.best_iteration = -1
        self.best_score = 100000

    def _append(self, rum_booster):
        """Add a booster to CVBooster."""
        self.rumboosters.append(rum_booster)

    def __getattr__(self, name):
        """Redirect methods call of CVBooster."""
        def handler_function(*args, **kwargs):
            """Call methods with each booster, and concatenate their results."""
            ret = []
            for rum_booster in self.rumboosters:
                for booster in rum_booster:
                    ret.append(getattr(booster, name)(*args, **kwargs))
                return ret
        return handler_function


def _make_n_folds(full_data, folds, nfold, params, seed, fpreproc=None, stratified=True,
                  shuffle=True, eval_train_metric=False, rum_structure=None):
    """Make a n-fold list of Booster from random indices."""
    full_data = full_data.construct()
    num_data = full_data.num_data()
    if folds is not None:
        if not hasattr(folds, '__iter__') and not hasattr(folds, 'split'):
            raise AttributeError("folds should be a generator or iterator of (train_idx, test_idx) tuples "
                                 "or scikit-learn splitter object with split method")
        if hasattr(folds, 'split'):
            group_info = full_data.get_group()
            if group_info is not None:
                group_info = np.array(group_info, dtype=np.int32, copy=False)
                flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            else:
                flatted_group = np.zeros(num_data, dtype=np.int32)
            folds = folds.split(X=np.empty(num_data), y=full_data.get_label(), groups=flatted_group)
    else:
        if any(params.get(obj_alias, "") in {"lambdarank", "rank_xendcg", "xendcg",
                                             "xe_ndcg", "xe_ndcg_mart", "xendcg_mart"}
               for obj_alias in _ConfigAliases.get("objective")):
            if not SKLEARN_INSTALLED:
                raise LightGBMError('scikit-learn is required for ranking cv')
            # ranking task, split according to groups
            group_info = np.array(full_data.get_group(), dtype=np.int32, copy=False)
            flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            group_kfold = _LGBMGroupKFold(n_splits=nfold)
            folds = group_kfold.split(X=np.empty(num_data), groups=flatted_group)
        elif stratified:
            if not SKLEARN_INSTALLED:
                raise LightGBMError('scikit-learn is required for stratified cv')
            skf = _LGBMStratifiedKFold(n_splits=nfold, shuffle=shuffle, random_state=seed)
            folds = skf.split(X=np.empty(num_data), y=full_data.get_label())
        else:
            if shuffle:
                randidx = np.random.RandomState(seed).permutation(num_data)
            else:
                randidx = np.arange(num_data)
            kstep = int(num_data / nfold)
            test_id = [randidx[i: i + kstep] for i in range(0, num_data, kstep)]
            train_id = [np.concatenate([test_id[i] for i in range(nfold) if k != i]) for k in range(nfold)]
            folds = zip(train_id, test_id)

    ret = CVRUMBooster()
    for train_idx, test_idx in folds:
        train_set = full_data.subset(sorted(train_idx))
        valid_set = full_data.subset(sorted(test_idx))
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            train_set, valid_set, tparam = fpreproc(train_set, valid_set, params.copy())
        else:
            tparam = params
        #create RUMBoosters with corresponding training, validation, and parameters sets
        cvbooster = RUMBooster()
        cvbooster.rum_structure = rum_structure
        reduced_valid_sets, name_valid_sets, is_valid_contain_train, train_data_name = cvbooster._preprocess_valids(train_set, params, valid_set)
        cvbooster._preprocess_data(train_set, reduced_valid_sets)
        cvbooster._preprocess_params(tparam)
        cvbooster._construct_boosters(train_data_name, is_valid_contain_train,
                                      name_valid_sets=name_valid_sets)

        ret._append(cvbooster)
    return ret


def _agg_cv_result(raw_results, eval_train_metric=False):
    """Aggregate cross-validation results."""
    cvmap = collections.OrderedDict()
    metric_type = {}
    for one_result in raw_results:
        for one_line in one_result:
            if eval_train_metric:
                key = f"{one_line[0]} {one_line[1]}"
            else:
                key = one_line[1]
            metric_type[key] = one_line[3]
            cvmap.setdefault(key, [])
            cvmap[key].append(one_line[2])
    return [('cv_agg', k, np.mean(v), metric_type[k], np.std(v)) for k, v in cvmap.items()]


def rum_cv(params, train_set, num_boost_round=100,
       folds=None, nfold=5, stratified=True, shuffle=True,
       metrics=None, fobj=None, feval=None, init_model=None,
       feature_name='auto', categorical_feature='auto',
       early_stopping_rounds=None, fpreproc=None,
       verbose_eval=None, show_stdv=True, seed=0,
       callbacks=None, eval_train_metric=False,
       return_cvbooster=False, rum_structure=None):
    """Perform the cross-validation with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for Booster.
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
        If generator or iterator, it should yield the train and test indices for each fold.
        If object, it should be one of the scikit-learn splitter classes
        (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
        and have ``split`` method.
        This argument has highest priority over other data split arguments.
    nfold : int, optional (default=5)
        Number of folds in CV.
    stratified : bool, optional (default=True)
        Whether to perform stratified sampling.
    shuffle : bool, optional (default=True)
        Whether to shuffle before splitting data.
    metrics : str, list of str, or None, optional (default=None)
        Evaluation metrics to be monitored while CV.
        If not None, the metric in ``params`` will be overridden.
    fobj : callable or None, optional (default=None)
        Customized objective function.
        Should accept two parameters: preds, train_data,
        and return (grad, hess).

            preds : list or numpy 1-D array
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            train_data : Dataset
                The training dataset.
            grad : list or numpy 1-D array
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of preds for each sample point.
            hess : list or numpy 1-D array
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of preds for each sample point.

        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i]
        and you should group grad and hess in this way as well.

    feval : callable, list of callable, or None, optional (default=None)
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, train_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : list or numpy 1-D array
                The predicted values.
                If ``fobj`` is specified, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            train_data : Dataset
                The training dataset.
            eval_name : str
                The name of evaluation function (without whitespace).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].
        To ignore the default metric corresponding to the used objective,
        set ``metrics`` to the string ``"None"``.
    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.
    feature_name : list of str, or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of str or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
    early_stopping_rounds : int or None, optional (default=None)
        Activates early stopping.
        CV score needs to improve at least every ``early_stopping_rounds`` round(s)
        to continue.
        Requires at least one metric. If there's more than one, will check all of them.
        To check only the first metric, set the ``first_metric_only`` parameter to ``True`` in ``params``.
        Last entry in evaluation history is the one from the best iteration.
    fpreproc : callable or None, optional (default=None)
        Preprocessing function that takes (dtrain, dtest, params)
        and returns transformed versions of those.
    verbose_eval : bool, int, or None, optional (default=None)
        Whether to display the progress.
        If True, progress will be displayed at every boosting stage.
        If int, progress will be displayed at every given ``verbose_eval`` boosting stage.
    show_stdv : bool, optional (default=True)
        Whether to display the standard deviation in progress.
        Results are not affected by this parameter, and always contain std.
    seed : int, optional (default=0)
        Seed used to generate the folds (passed to numpy.random.seed).
    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    eval_train_metric : bool, optional (default=False)
        Whether to display the train metric in progress.
        The score of the metric is calculated again after each training step, so there is some impact on performance.
    return_cvbooster : bool, optional (default=False)
        Whether to return Booster models trained on each fold through ``CVBooster``.
    rum_structure : dict, optional (default=None)
        List of dictionaries specifying the RUM structure. 
        The list must contain one dictionary for each class, which describes the 
        utility structure for that class. 
        Each dictionary has three allowed keys. 
            'cols': list of columns included in that class
            'monotone_constraints': list of monotonic constraints on parameters
            'interaction_constraints': list of interaction constraints on features

    Returns
    -------
    eval_hist : dict
        Evaluation history.
        The dictionary has the following format:
        {'metric1-mean': [values], 'metric1-stdv': [values],
        'metric2-mean': [values], 'metric2-stdv': [values],
        ...}.
        If ``return_cvbooster=True``, also returns trained boosters via ``cvbooster`` key.
    """
    if not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")

    params = copy.deepcopy(params)
    if fobj is not None:
        for obj_alias in _ConfigAliases.get("objective"):
            params.pop(obj_alias, None)
        params['objective'] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            _log_warning(f"Found `{alias}` in params. Will use it instead of argument")
            num_boost_round = params.pop(alias)
    params["num_iterations"] = num_boost_round
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        _log_warning("'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. "
                     "Pass 'early_stopping()' callback via 'callbacks' argument instead.")
    for alias in _ConfigAliases.get("early_stopping_round"):
        if alias in params:
            early_stopping_rounds = params.pop(alias)
    params["early_stopping_round"] = early_stopping_rounds
    first_metric_only = params.get('first_metric_only', False)

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    if isinstance(init_model, (str, Path)):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    else:
        predictor = None

    if metrics is not None:
        for metric_alias in _ConfigAliases.get("metric"):
            params.pop(metric_alias, None)
        params['metric'] = metrics

    train_set._update_params(params) \
             ._set_predictor(predictor) \
             .set_feature_name(feature_name) \
             .set_categorical_feature(categorical_feature)

    results = collections.defaultdict(list)
    cvfolds = _make_n_folds(train_set, folds=folds, nfold=nfold,
                            params=params, seed=seed, fpreproc=fpreproc,
                            stratified=stratified, shuffle=shuffle,
                            eval_train_metric=eval_train_metric, rum_structure=rum_structure)

    # setup callbacks
    if callbacks is None:
        callbacks = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks = set(callbacks)
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.add(callback.early_stopping(early_stopping_rounds, first_metric_only, verbose=False))
    if verbose_eval is not None:
        _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
                     "Pass 'log_evaluation()' callback via 'callbacks' argument instead.")
    if verbose_eval is True:
        callbacks.add(callback.log_evaluation(show_stdv=show_stdv))
    elif isinstance(verbose_eval, int):
        callbacks.add(callback.log_evaluation(verbose_eval, show_stdv=show_stdv))

    callbacks_before_iter = {cb for cb in callbacks if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter = callbacks - callbacks_before_iter
    callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter, key=attrgetter('order'))

    for i in range(num_boost_round):
        cross_ent = []
        raw_results = []
        #train all rumboosters
        for rumbooster in cvfolds.rumboosters:
            rumbooster._preds = rumbooster._inner_predict()
            for j, booster in enumerate(rumbooster.boosters):
                for cb in callbacks_before_iter:
                    cb(callback.CallbackEnv(model=booster,
                                            params=rumbooster.params[j],
                                            iteration=i,
                                            begin_iteration=0,
                                            end_iteration=num_boost_round,
                                            evaluation_result_list=None))
                rumbooster._current_j = j
                booster.update(train_set = rumbooster.train_set[j], fobj=rumbooster.f_obj)

            valid_sets = rumbooster.valid_sets
            for valid_set in valid_sets:
                preds_valid = rumbooster._inner_predict(data = valid_set)
                raw_results.append(preds_valid)
                cross_ent.append(rumbooster.cross_entropy(preds_valid, valid_set[0].get_label().astype(int)))

        results[f'Cross entropy --- mean'].append(np.mean(cross_ent))
        results[f'Cross entropy --- stdv'].append(np.std(cross_ent))
        if verbose_eval and i % 10 == 0:
            print('[{}] -- Cross entropy mean: {}, with std: {}'.format(i + 1, np.mean(cross_ent), np.std(cross_ent)))
        
        if np.mean(cross_ent) < cvfolds.best_score:
            cvfolds.best_score = np.mean(cross_ent)
            cvfolds.best_iteration = i + 1 

        if early_stopping_rounds is not None and cvfolds.best_iteration + early_stopping_rounds < i+1:
            print('Early stopping at iteration {} with a cross entropy best score of {}'.format(cvfolds.best_iteration,cvfolds.best_score))
            for k in results:
                results[k] = results[k][:cvfolds.best_iteration]
            break
        #res = _agg_cv_result(raw_results, eval_train_metric)
        #try:
        #    for cb in callbacks_after_iter:
        #        cb(callback.CallbackEnv(model=cvfolds,
        #                                params=params,
        #                                iteration=i,
        #                                begin_iteration=0,
        #                                end_iteration=num_boost_round,
        #                                evaluation_result_list=res))
        #except callback.EarlyStopException as earlyStopException:
        #    cvfolds.best_iteration = earlyStopException.best_iteration + 1
        #    for k in results:
        #        results[k] = results[k][:cvfolds.best_iteration]
        #    break

    if return_cvbooster:
        results['cvbooster'] = cvfolds

    return dict(results)