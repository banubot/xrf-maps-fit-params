'''
08 Tune and train the xgboost model
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pickle

spectra_normalized = pd.read_csv('./training_data/07_train_int_spectra.csv', header=None)
fit_params_standard = pd.read_csv('./training_data/07_train_fit_params_standard.csv')
fit_params_constant = pd.read_csv('./training_data/04_fit_params_constant.csv')

print(len(spectra_normalized))
print(len(fit_params_standard))

x_train = spectra_normalized
y_train = fit_params_standard.drop(columns=fit_params_constant.columns)

# Create and train the XGBoost Regressor
regressor = xgb.XGBRegressor()

# Define hyperparameters to search
params = {
    "eta": [0.2, 0.3, 0.4],
    "gamma": [0, 1, 2],
    "max_depth": [5, 6, 7],
    "lambda": [0, 1, 2],
    "alpha": [0, 1],
}

search = GridSearchCV(regressor, params, n_jobs=32, verbose=True, cv=3)
search.fit(x_train, y_train)
print(search.best_params_)

model = search.best_estimator_
# Save the model to disk
filename = '08_xgboost.sav'
pickle.dump(model, open(filename, 'wb'))