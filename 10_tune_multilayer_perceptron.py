'''
10 Tune and train the multilayer perceptron model
'''
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import pickle

spectra_normalized = pd.read_csv('./training_data/08_train_int_spectra.csv', header=None)
fit_params_standard = pd.read_csv('./training_data/07_train_fit_params_standard.csv')
fit_params_constant = pd.read_csv('./training_data/04_fit_params_constant.csv')

print(len(spectra_normalized))
print(len(fit_params_standard))

x_train = spectra_normalized
y_train = fit_params_standard.drop(columns=fit_params_constant.columns)

# Create and train the MLP Regressor
regressor = MLPRegressor(early_stopping=True)

# Define hyperparameters to search
params = {
    "batch_size": [10, 50, 100],
    "alpha": [0.0001, 0.001, 0.00001],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "learning_rate_init": [0.01, 0.001, 0.0001],
    "max_iter": [500, 1000, 5000],
}

search = GridSearchCV(regressor, params, n_jobs=-1, verbose=True, cv=5)
search.fit(x_train, y_train)
print(search.best_params_)

model = search.best_estimator_
# Save the best model to disk
filename = '10_multilayer_perceptron.sav'
pickle.dump(model, open(filename, 'wb'))