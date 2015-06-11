import time
from itertools import chain, combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from joblib import Parallel, delayed  
import multiprocessing
import numpy as np
from pprint import pprint
import testga
import pickle
import argparse

knowledge_base_initial_size = 25

meta_data_path = 'meta_features.csv'
meta_data_header_size = 1
meta_data_delimiter = ','
meta_data_used_cols = [x for x in range(1,13)]

elitism = 0

names_column = 0

x_input_path = "normalized_X.pkl"
y_input_path = "normalized_Y.pkl"

X_all = None
y_all = None

optimal_parameters_path = 'optimal_parameters_all.csv'
optimal_parameters_header_size = 1
optimal_parameters_delimiter = ','
optimal_parameters_used_cols = [1,2]

ap = argparse.ArgumentParser()
ap.add_argument('-e', action='store_true')
args = ap.parse_args()

if (args.e):
    print("Elitism")
    elitism = 1
else:
    print("No elitism")

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

def meta_feature_selection(meta_data, optimal_parameters, k=4):
	features = [i for i in range(meta_data.shape[1])]
	datasets = [i for i in range(meta_data.shape[0])]
	min_distance = float("inf")
	selected_meta_features = None

	for subset in powerset(features):
		meta_data_subset = meta_data[:,subset]
		# uses k+1 because we are getting the distances from all to all, and the closest is itself = 0
		nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(meta_data_subset)

		optimal_parameters_distance = 0
		for ds in datasets:
			neighbors_distances, neighbors_indices = nbrs.kneighbors(meta_data_subset[ds,:])

			ds_optimal_parameters = optimal_parameters[ds,:].reshape( (1,optimal_parameters.shape[1]) ) 
			neighbors_optimal_parameters = optimal_parameters[neighbors_indices,:].reshape( (k,optimal_parameters.shape[1]) ) 

			optimal_parameters_distance += np.sum(euclidean_distances(ds_optimal_parameters, neighbors_optimal_parameters))

		#print ("actual=%.4f, min=%.4f, meta_features=%s" % (optimal_parameters_distance, min_distance, str(subset)))
		if optimal_parameters_distance < min_distance:
			min_distance = optimal_parameters_distance
			selected_meta_features = subset
		
	return selected_meta_features

def generate_initial_population(knowledge_base, optimal_parameters, dataset_meta_features, n_individuals=4):
	nbrs = NearestNeighbors(n_neighbors=n_individuals, algorithm='brute').fit(knowledge_base)
	dataset_meta_features = dataset_meta_features.reshape( (1, knowledge_base.shape[1]) )
	distances, indices = nbrs.kneighbors(dataset_meta_features)

	initial_population = optimal_parameters[indices,:]

	return initial_population.reshape( (n_individuals, optimal_parameters.shape[1]) )

meta_data = np.genfromtxt(meta_data_path, dtype=np.float, usecols=meta_data_used_cols,delimiter=meta_data_delimiter,skip_header=meta_data_header_size)
knowledge_base_max_size = meta_data.shape[0]

names = np.genfromtxt(meta_data_path, dtype='U', usecols=names_column,delimiter=meta_data_delimiter,skip_header=meta_data_header_size)
print("Knowlodge base size: %d" % knowledge_base_initial_size)

optimal_parameters = np.genfromtxt(optimal_parameters_path, dtype=np.float, usecols=optimal_parameters_used_cols,delimiter=optimal_parameters_delimiter,skip_header=optimal_parameters_header_size)

# Loading datasets
with open(x_input_path, 'rb') as f:
    X_all = pickle.load(f)

with open(y_input_path, 'rb') as f:
    y_all = pickle.load(f)


print("Selecting meta features...")
selected_features = meta_feature_selection(meta_data[0:knowledge_base_initial_size,:], optimal_parameters[0:knowledge_base_initial_size,:], k=5)


print("Selected features: %s" % str(selected_features))

kb_size = knowledge_base_initial_size

ini = time.time()

for ds in range(knowledge_base_initial_size, knowledge_base_max_size):
	name = names[ds]
	print(name)

	X = X_all[name]
	y = y_all[name]

	kb = meta_data[0:kb_size,:]
	kb_optimal_parameters = optimal_parameters[0:kb_size,:]

	kb = kb[:,selected_features]
	dataset_meta_features = meta_data[ds, selected_features]

	initial_population = generate_initial_population(kb, kb_optimal_parameters, dataset_meta_features, n_individuals=6)

	#genetic algorithm
	optimal_parameters[ds,:], accuracy = testga.run(X, y, initial_population,  pop_size=len(initial_population), elite=elitism)
	print("Optimal parameters: %s\nAccuracy: %f" % (str(optimal_parameters[ds,:]), accuracy) )

	kb_size += 1

end = time.time()

duration = time.strftime("%H:%M:%S", time.gmtime(end - ini))

print("Duration: %s" % duration)
