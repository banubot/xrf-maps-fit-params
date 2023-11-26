'''
08 Tune and train the random forest model
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle

spectra_normalized = pd.read_csv('./training_data/07_train_int_spectra.csv', header=None)
fit_params_standard = pd.read_csv('./training_data/07_train_fit_params_standard.csv')
fit_params_constant = pd.read_csv('./training_data/04_fit_params_constant.csv')

print(len(spectra_normalized))
print(len(fit_params_standard))

x_train = spectra_normalized
y_train = fit_params_standard.drop(columns=fit_params_constant.columns)

# Create and train the Random Forest Regressor
regressor = RandomForestRegressor()

# Define hyperparameters to search
params = {
    "n_estimators": [50, 100, 150, 500],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 3, 4],
}

search = GridSearchCV(regressor, params, n_jobs=16, verbose=True, cv=3)
search.fit(x_train, y_train)
print(search.best_params_)

model = search.best_estimator_
# Save the best model to disk
filename = '08_random_forest.sav'
pickle.dump(model, open(filename, 'wb'))