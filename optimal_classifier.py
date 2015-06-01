from __future__ import print_function
import argparse

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle
from joblib import Parallel, delayed  
import multiprocessing

output_path = "optimal_parameters.csv"
x_input_path = "normalized_X.pkl"
y_input_path = "normalized_Y.pkl"
probe_prefix = "PROBE_"
ap = argparse.ArgumentParser()
ap.add_argument('-p', action='store_true')
ap.add_argument('--singlecore', action='store_true')
args = ap.parse_args()

if (args.p):
    x_input_path = probe_prefix + x_input_path
    y_input_path = probe_prefix + y_input_path

def optimal_training(dataset_name,X,y):
    # Set the parameters by cross-validation
    # rbf = radial basis function
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [pow(2,x) for x in range(-10,5)], 'C': [pow(2,x) for x in range(-2,13)]}]

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10, scoring='accuracy')
    clf.fit(X, y)

    #print("Best parameters set found on development set:")
    #print(clf.best_estimator_)
    with open('../uci/optimal_' + dataset_name, "w") as optm_file:
        optm_file.write(str(clf.best_params_))
        optm_file.write("accuracy: %d" % clf.best_score_)

    return (dataset_name, clf.best_params_, clf.best_score_)


# Loading datasets
with open(x_input_path, 'rb') as f:
    X_all = pickle.load(f)

with open(y_input_path, 'rb') as f:
    y_all = pickle.load(f)

if not args.singlecore:
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=4)(delayed(optimal_training)(dataset,X_all[dataset], y_all[dataset]) for dataset in X_all.keys())
else:
    for dataset in X_all.keys():
        optimal_training(dataset,X_all[dataset], y_all[dataset])

with open(output_path, "w") as output_file:
    output_file.write("dataset_name,gamma,C,accuracy\n")
    for dataset_name, params, accuracy in results:
        output_file.write("%s,%f,%f,%f\n" % (dataset_name, params['gamma'], params['C'], accuracy))

    # Note the problem is too easy: the hyperparameter plateau is too flat and the
    # output model is the same for precision and recall with ties in quality.
