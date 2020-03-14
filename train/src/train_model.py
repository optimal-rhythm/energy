import numpy as np
import pandas as pd
import sklearn
import pickle
import joblib
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from util import *

# Read in the data
energy = pd.read_excel('/app/ENB2012_data.xlsx').dropna()
 
# Rename the features
energy.columns = ["RC", "SA", "WA", "RA", "OH", "OT", "GA", "GAD", "HL", "CL"]

# Categorical vs numerical features
features, labels = list(energy)[:8], list(energy)[8:]
numerical_features, categorical_features = features[:5]+['GA'], ['OT', 'GAD']
categories = ['none', 'uniform', 'north', 'east', 'south', 'west']
transformer = lambda k: categories[int(k)]
energy[categorical_features] = energy[categorical_features].applymap(transformer)
energy['OT'] = energy['OT'].astype('category')
energy['GAD'] = energy['GAD'].astype('category')

# Split into test data and training data
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(energy, energy['RC']):
    train_set = energy.loc[train_index]
    train_set.index = range(614)
    test_set = energy.loc[test_index]
 
# Get only the training data set
energy_features, energy_labels = train_set[features], train_set[labels]
energy_num_feats, energy_cat_feats = energy_features[numerical_features], energy_features[categorical_features]

# Build a pipeline to reuse
numerical_pipeline = Pipeline([
        ('selector', DataFrameSelector(numerical_features)),
        ('std_scaler', StandardScaler()),
    ])
 
categorical_pipeline = Pipeline([
        ('selector', DataFrameSelector(categorical_features)),
        ('cat_encoder', CategoricalEncoder()),
    ])
 
final_pipeline = FeatureUnion(transformer_list=[
        ("numerical_pipeline", numerical_pipeline),
        ("categorical_pipeline", categorical_pipeline),
    ])
 
energy_processed = final_pipeline.fit_transform(energy_features) 
regressors = {'HL': {}, 'CL': {}}
columns = ["Training RMSE", "Cross Validation RMSE", "Cross Validation SD"]
tables = {'HL': pd.DataFrame(columns=columns), 'CL': pd.DataFrame(columns=columns)}

# Use RandomForest as the regression model algo
forest_reg = RandomForestRegressor(random_state=42)
param_grid = [
    # 3 x 4 combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # 2 x 3 combinations with bootstrap = false
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

# Hyper-parameter grid search
for label in energy_labels:
    target = energy_labels[label]
    if param_grid is None:
        best_regressor = forest_reg
    else:
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(energy_processed, target)
        best_regressor = grid_search.best_estimator_
    best_regressor.fit(energy_processed, target)
    predictions = best_regressor.predict(energy_processed)
    rmse = np.sqrt(mean_squared_error(target, predictions))
    scores = cross_val_score(best_regressor, energy_processed, target, scoring="neg_mean_squared_error", cv=10)
    scores = np.sqrt(-scores)
    regressors[target.name]["Random Forest"] = best_regressor
    values = dict(zip(columns, [rmse, scores.mean(), scores.std()]))
    row = pd.DataFrame(values, index=["Random Forest"], columns=columns)
    table = tables[target.name].append(row)
    tables[target.name] = table[~table.index.duplicated(keep='last')]

# Print metrics
print("============")
print("Heating Load")
print(tables['HL'])
print("============")
print("Cooling Load")
print(tables['CL'])
print("============")

# Pickle the results
final_models = {label: regressors[label]["Random Forest"] for label in labels}
pickle.dump(final_models['HL'], open('/app/HL_model.pickle', 'wb'))
pickle.dump(final_models['CL'], open('/app/CL_model.pickle', 'wb'))
joblib.dump(final_pipeline, '/app/model.joblib')