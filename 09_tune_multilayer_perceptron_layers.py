'''
09 Tune and train the multilayer perceptron model layer numbers and sizes
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
    "hidden_layer_sizes": [(100), (100, 100), (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100),
    (50,100,150,200), (500,100,100,500), (100,200,300,400),(25,50,75,100),(10,20,30,40)],
}

search = GridSearchCV(regressor, params, n_jobs=-1, verbose=True, cv=5)
search.fit(x_train, y_train)
print(search.best_params_)

model = search.best_estimator_
# Save the best model to disk
filename = '09_multilayer_perceptron.sav'
pickle.dump(model, open(filename, 'wb'))