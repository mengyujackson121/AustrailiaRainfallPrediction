import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import code.preprocessing as prep
import code.visualization as vis
from sklearn.model_selection import train_test_split
from sklearn import datasets, ensemble, linear_model, neural_network
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import RFECV

# NEW
from sklearn.inspection import plot_partial_dependence
from xgboost import XGBRegressor

from sklearn.decomposition import PCA

pipe = Pipeline([('scaler', StandardScaler()), ('nn', neural_network.MLPRegressor(solver="lbfgs", random_state = RANDOM_SEED))])


# {"Clean Data: <class 'sklearn.ensemble._forest.RandomForestRegressor'>": 0.8774411263178115,
#  "Clean Data: <class 'sklearn.ensemble._gb.GradientBoostingRegressor'>": 0.8632543928275058,
#  "Clean Data: <class 'sklearn.linear_model._base.LinearRegression'>": 0.7257411230925401,
#  "Clean Data: <class 'sklearn.linear_model._bayes.BayesianRidge'>": 0.725527984428523,
#  "Clean Data: <class 'sklearn.linear_model._ridge.Ridge'>": 0.7255757950042747,
#  "Clean Data: <class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>": 0.5685389218888475,
#  "Clean Data: <class 'sklearn.pipeline.Pipeline'>": 0.8608797017030435,
#  "Cyclic Month Data: <class 'sklearn.ensemble._forest.RandomForestRegressor'>": 0.876507316849388,
#  "Cyclic Month Data: <class 'sklearn.ensemble._gb.GradientBoostingRegressor'>": 0.866048321867755,
#  "Cyclic Month Data: <class 'sklearn.linear_model._base.LinearRegression'>": 0.7254677621820269,
#  "Cyclic Month Data: <class 'sklearn.linear_model._bayes.BayesianRidge'>": 0.725243866660823,
#  "Cyclic Month Data: <class 'sklearn.linear_model._ridge.Ridge'>": 0.7253016703243023,
#  "Cyclic Month Data: <class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>": 0.5586926522991034,
#  "Cyclic Month Data: <class 'sklearn.pipeline.Pipeline'>": 0.8478080755692423}

scores = {}
modles = [linear_model.Ridge(random_state = RANDOM_SEED),
          linear_model.BayesianRidge(),
          linear_model.LinearRegression(),
          ensemble.RandomForestRegressor(random_state = RANDOM_SEED),
          ensemble.GradientBoostingRegressor(random_state = RANDOM_SEED),
          neural_network.MLPRegressor(solver="lbfgs", random_state = RANDOM_SEED),
          pipe]

for modle in modles:
    scores[f"Clean Data: {type(modle)}"] = prep.model(modle, X_train, X_test, y_train, y_test)
    scores[f"Cyclic Month Data: {type(modle)}"] = prep.model(modle, Xc_train, Xc_test, yc_train, yc_test)
pprint.pprint(scores)



# 0.8399629316348763
nn_2 = Pipeline([('scaler', StandardScaler()), ('nn', neural_network.MLPRegressor(solver="lbfgs", random_state = RANDOM_SEED, max_iter=100000))])
prep.model(nn_2, X_train, X_test, y_train, y_test)


# 0.29004453261657537
nn_3 = Pipeline([('scaler', StandardScaler()), ('nn', neural_network.MLPRegressor(solver="lbfgs", random_state = RANDOM_SEED, max_iter=20000, hidden_layer_sizes=(100,40,10)))])
prep.model(nn_3, X_train, X_test, y_train, y_test)


# 0.8829413723729465
xg_model= XGBRegressor(random_state= RANDOM_SEED)
prep.model(xg_model, X_train, X_test, y_train, y_test)


# 0.9005361799583668
xg_model2= XGBRegressor(learning_rate=0.008, max_depth=6, gamma=0, n_estimators=4000, random_state= RANDOM_SEED)
prep.model(xg_model2, X_train, X_test, y_train, y_test)


# 0.9003123683320535
xg_model3= XGBRegressor(learning_rate=0.03, max_depth=6, gamma=0, n_estimators=4000, random_state= RANDOM_SEED)
prep.model(xg_model3, X_train, X_test, y_train, y_test)


#0.894989546861814
xg_model4= XGBRegressor(learning_rate=0.02, max_depth=8, gamma=0, n_estimators=10000, random_state= RANDOM_SEED)
prep.model(xg_model4, X_train, X_test, y_train, y_test)


# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.03, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=4000, n_jobs=16, num_parallel_tree=1, random_state=5,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# pca_xg_model = Pipeline([('pca', PCA()),
#                         ('xgboost', XGBRegressor(random_state=RANDOM_SEED))])
pca = PCA()
pca.fit(X_train)
X_train_trans = pca.transform(X_train)
X_test_trans = pca.transform(X_test)
xgb_model5 = XGBRegressor(learning_rate=0.03, max_depth=6, gamma=0, n_estimators=4000, random_state= RANDOM_SEED)
xgb_model5.fit(X_train_trans, y_train)
#pca_xg_model.fit(X_train)


# 0.8369439865627064
xgb_model5.score(X_test_trans, y_test)


# 0.8968396958133438
xg_model6= XGBRegressor(learning_rate=0.03, max_depth=4, gamma=0, n_estimators=5000, random_state= RANDOM_SEED)
prep.model(xg_model6, X_train, X_test, y_train, y_test)


# 0.8974668268967836
# grid_params = [
#     {'n_estimators': [500, 1000, 3000, 5000],
#      'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, .1],
#      'max_depth': [3, 4, 5, 6, 7],
#      'gamma': [0, 1, 5]}
# ]
grid_params = [
    {'n_estimators': [4000, 6000],
     'learning_rate': [ 0.02],
     'max_depth': [10],
     'gamma': [0]}
]
grid_search2 = GridSearchCV(XGBRegressor(random_state= RANDOM_SEED), grid_params, cv=KFold(5, shuffle=True, random_state=RANDOM_SEED))
grid_search2.fit(X_train, y_train)
grid_search2.score(X_test, y_test)
#prep.model(xg_model3, X_train, X_test, y_train, y_test)

# best so far:
# .8997
# {'gamma': 0, 'learning_rate': 0.03, 'max_depth': 4, 'n_estimators': 5000}
print(X_train.columns)
print(grid_search2.best_estimator_.feature_importances_)

# Index(['id', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
#       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
#       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
#       'sqft_living15', 'sqft_lot15', 'year', 'day', 'month_sin', 'month_cos',
#       'sqft_living_dif', 'sqft_lot_dif'],
#      dtype='object')
# [0.00243093 0.00209524 0.00515083 0.15690187 0.00429884 0.00290446
# 0.24368031 0.03870978 0.00883588 0.35904017 0.01091988 0.00305385
# 0.01780545 0.00596826 0.01348273 0.05392285 0.03054818 0.01740094
# 0.00293573 0.00739305 0.0025103  0.00267542 0.00206679 0.00314914
# 0.00211913]

polynomial_regression = Pipeline([('pca', PCA()), ("polynomial_features", PolynomialFeatures(interaction_only=True)), ("linear_regression", linear_model.LinearRegression())])
X_train_poly = PolynomialFeatures(interaction_only=True).fit_transform(X_train)

rfe_linear_model = RFECV(linear_model.LinearRegression(),
                         step=1,
                         cv=5)
rfe_linear_model.fit(X_train_poly, y_train)
#polynomial_regression.fit(X_train, y_train)

#polynomial_regression.params_



# GridSearchCV(cv=KFold(n_splits=5, random_state=5, shuffle=True),
#              estimator=GradientBoostingRegressor(random_state=5),
#              param_grid=[{'learning_rate': [0.08, 0.1, 0.12],
#                           'loss': ['ls', 'lad', 'huber', 'quantile'],
#                           'max_depth': [3, 5, 7], 'n_estimators': [1000]}])

grid_params = [
    {'n_estimators': [1000], 'learning_rate': [0.08, .1, .12], 'loss': ['ls', 'lad', 'huber', 'quantile'], 'max_depth': [3, 5, 7]}
]

grid_search = GridSearchCV(ensemble.GradientBoostingRegressor(random_state = RANDOM_SEED), grid_params, cv=KFold(5, shuffle=True, random_state=RANDOM_SEED))
grid_search.fit(X_train, y_train)


