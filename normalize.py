import os
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import pickle

def parseCategoricalMissingValues(data, missing_values="?"):
	
	if len(np.shape(data)) == 1:
		unique,pos = np.unique(data,return_inverse=True) #Finds all unique elements and their positions
		counts = np.bincount(pos)                     #Count the number of each unique element
		maxpos = counts.argmax()
		most_freq = unique[maxpos]
		data = np.core.defchararray.replace(data, missing_values, most_freq)
	else:	
		m = data.shape[1]
		for j in range(m):
			unique,pos = np.unique(data[:,j],return_inverse=True) #Finds all unique elements and their positions
			counts = np.bincount(pos)                     #Count the number of each unique element
			maxpos = counts.argmax()
			most_freq = unique[maxpos]
			data[:,j] = np.core.defchararray.replace(data[:,j], missing_values, most_freq)

numerical_features = dict()
categorical_features = dict()
classes = dict()
normalized_features = dict()
normalized_classes = dict()
delimiters = dict()
missing = dict()
header = dict()

x_output_path = "normalized_X.pkl"
y_output_path = "normalized_Y.pkl"
base_dir = "../uci/original"
out_dir = "../uci/normalized"
meta_data_path = "meta_data.csv"
meta_data_lines = []

#setup databases parameters
numerical_features['abalone.data'] = [x for x in range(1,8)]
categorical_features['abalone.data'] = [0]
classes['abalone.data'] = [8]
delimiters['abalone.data'] = ","
header['abalone.data'] = 0

numerical_features['Amazon_initial_50_30_10000.arff'] = [x for x in range(0,10000)]
categorical_features['Amazon_initial_50_30_10000.arff'] = []
classes['Amazon_initial_50_30_10000.arff'] = [10000]
delimiters['Amazon_initial_50_30_10000.arff'] = ","
header['Amazon_initial_50_30_10000.arff'] = 10005

numerical_features['anneal.data'] = [3,4,7,8,12,32,33,34,37]
categorical_features['anneal.data'] = [0,1,2,5,6,9,10,11] + [x for x in range(13,32)] + [35,36]
classes['anneal.data'] = [38]
delimiters['anneal.data'] = ","
missing['anneal.data'] = "?"
header['anneal.data'] = 0

numerical_features['arrhythmia.data'] = [0] + [x for x in range(2,21)] + [x for x in range(27,33)] + [x for x in range(39,45)] + [x for x in range(51,57)] + [x for x in range(63,69)] + [x for x in range(75,81)] + [x for x in range(87,93)] + [x for x in range(99,105)] + [x for x in range(111,117)] + [x for x in range(123,129)] + [x for x in range(135,141)] + [x for x in range(147,153)] + [x for x in range(159,279)]
categorical_features['arrhythmia.data'] = [1] + [x for x in range(21,27)] + [x for x in range(33,39)] + [x for x in range(45,51)] + [x for x in range(57,63)] + [x for x in range(69,75)] + [x for x in range(81,87)] + [x for x in range(93,99)] + [x for x in range(105,111)] + [x for x in range(117,123)] + [x for x in range(129,135)] + [x for x in range(141,147)] + [x for x in range(153,159)]
classes['arrhythmia.data'] = [279]
delimiters['arrhythmia.data'] = ","
missing['arrhythmia.data'] = "?"
header['arrhythmia.data'] = 0

numerical_features['audiology.standardized.data'] = []
categorical_features['audiology.standardized.data'] = [x for x in range(0,69)]
classes['audiology.standardized.data'] = [70]
delimiters['audiology.standardized.data'] = ","
header['audiology.standardized.data'] = 0

numerical_features['bridges.data.version1'] = [2,3,5,6]
categorical_features['bridges.data.version1'] = [1,4,7,8,9,10,11]
classes['bridges.data.version1'] = [12]
delimiters['bridges.data.version1'] = ","
missing['bridges.data.version1'] = "?"
header['bridges.data.version1'] = 0

numerical_features['bridges.data.version2'] = [2,6]
categorical_features['bridges.data.version2'] = [1,3,4,5,7,8,9,10,11]
classes['bridges.data.version2'] = [12]
delimiters['bridges.data.version2'] = ","
missing['bridges.data.version2'] = "?"
header['bridges.data.version2'] = 0

numerical_features['car.data'] = []
categorical_features['car.data'] = [x for x in range(0,6)]
classes['car.data'] = [6]
delimiters['car.data'] = ","
header['car.data'] = 0

numerical_features['cmc.data'] = [0,3]
categorical_features['cmc.data'] = [1,2,4,5,6,7,8]
classes['cmc.data'] = [9]
delimiters['cmc.data'] = ","
header['cmc.data'] = 0

numerical_features['CNAE-9.data'] = [x for x in range(1,857)]
categorical_features['CNAE-9.data'] = []
classes['CNAE-9.data'] = [0]
delimiters['CNAE-9.data'] = ","
header['CNAE-9.data'] = 0

numerical_features['connect-4.data'] = []
categorical_features['connect-4.data'] = [x for x in range(0,42)]
classes['connect-4.data'] = [42]
delimiters['connect-4.data'] = ","
header['connect-4.data'] = 0

numerical_features['covtype.data'] = [x for x in range(0,10)]
categorical_features['covtype.data'] = [10,11]
classes['covtype.data'] = [12]
delimiters['covtype.data'] = ","
header['covtype.data'] = 0

numerical_features['dermatology.data'] = [x for x in range(0,10)] + [x for x in range(11,34)]
categorical_features['dermatology.data'] = [10]
classes['dermatology.data'] = [34]
delimiters['dermatology.data'] = ","
missing['dermatology.data'] = "?"
header['dermatology.data'] = 0

numerical_features['ecoli.data'] = [x for x in range(1,8)]
categorical_features['ecoli.data'] = []
classes['ecoli.data'] = [8]
delimiters['ecoli.data'] = None
header['ecoli.data'] = 0

numerical_features['fertility_Diagnosis.txt'] = [1,6,7,8]
categorical_features['fertility_Diagnosis.txt'] = [0,2,3,4,5]
classes['fertility_Diagnosis.txt'] = [9]
delimiters['fertility_Diagnosis.txt'] = ","
header['fertility_Diagnosis.txt'] = 0

numerical_features['flag.data'] = [3,4,7,8,9] + [x for x in range(18,23)]
categorical_features['flag.data'] = [1,2,5] + [x for x in range(10,18)] + [x for x in range(23,30)]
classes['flag.data'] = [6]
delimiters['flag.data'] = ","
header['flag.data'] = 0

numerical_features['glass.data'] = [x for x in range(1,10)]
categorical_features['glass.data'] = []
classes['glass.data'] = [10]
delimiters['glass.data'] = ","
header['glass.data'] = 0

numerical_features['hepatitis.data'] = [14]
categorical_features['hepatitis.data'] = [x for x in range(1,14)] + [x for x in range(15,20)]
classes['hepatitis.data'] = [0]
delimiters['hepatitis.data'] = ","
missing['hepatitis.data'] = "?"
header['hepatitis.data'] = 0

numerical_features['house-votes-84.data'] = []
categorical_features['house-votes-84.data'] = [x for x in range(1,14)] + [x for x in range(2,17)]
classes['house-votes-84.data'] = [0]
delimiters['house-votes-84.data'] = ","
missing['house-votes-84.data'] = "?"
header['house-votes-84.data'] = 0

numerical_features['ionosphere.data'] = [x for x in range(0,34)]
categorical_features['ionosphere.data'] = []
classes['ionosphere.data'] = [34]
delimiters['ionosphere.data'] = ","
header['ionosphere.data'] = 0

numerical_features['iris.data'] = [x for x in range(0,4)]
categorical_features['iris.data'] = []
classes['iris.data'] = [4]
delimiters['iris.data'] = ","
header['iris.data'] = 0

numerical_features['krkopt.data'] = [1,3,5]
categorical_features['krkopt.data'] = [0,2,4]
classes['krkopt.data'] = [6]
delimiters['krkopt.data'] = ","
header['krkopt.data'] = 0

numerical_features['lung-cancer.data'] = []
categorical_features['lung-cancer.data'] = [x for x in range(1,57)]
classes['lung-cancer.data'] = [0]
delimiters['lung-cancer.data'] = ","
missing['lung-cancer.data'] = "?"
header['lung-cancer.data'] = 0

numerical_features['mammographic_masses.data'] = [0,1,4]
categorical_features['mammographic_masses.data'] = [2,3]
classes['mammographic_masses.data'] = [5]
delimiters['mammographic_masses.data'] = ","
missing['mammographic_masses.data'] = "?"
header['mammographic_masses.data'] = 0

numerical_features['pop_failures.dat'] = [x for x in range(2,20)]
categorical_features['pop_failures.dat'] = []
classes['pop_failures.dat'] = [20]
delimiters['pop_failures.dat'] = None
header['pop_failures.dat'] = 1

numerical_features['tictactoe.data'] = []
categorical_features['tictactoe.data'] = [x for x in range(0,9)]
classes['tictactoe.data'] = [9]
delimiters['tictactoe.data'] = ","
header['tictactoe.data'] = 0

numerical_features['trains.data'] = [0,1,2,5,7,10,12,15,17,20]
categorical_features['trains.data'] = [3,4,6,8,9,11,13,14,16,18,19,21] + [x for x in range(22,32)]
classes['trains.data'] = [32]
delimiters['trains.data'] = " "
missing['trains.data'] = "-"
header['trains.data'] = 0

numerical_features['transfusion.data'] = [0,1,2,3]
categorical_features['transfusion.data'] = []
classes['transfusion.data'] = [4]
delimiters['transfusion.data'] = ","
header['transfusion.data'] = 1

numerical_features['wine.data'] = [x for x in range(1,13)]
categorical_features['wine.data'] = []
classes['wine.data'] = [0]
delimiters['wine.data'] = ","
header['wine.data'] = 0

numerical_features['zoo.data'] = [13]
categorical_features['zoo.data'] = [x for x in range(1,13)] + [x for x in range(14,17)]
classes['zoo.data'] = [17]
delimiters['zoo.data'] = ","
header['zoo.data'] = 0


names = ['abalone.data', 'Amazon_initial_50_30_10000.arff', 'anneal.data', 'arrhythmia.data', 'audiology.standardized.data', 'bridges.data.version1', 'bridges.data.version2', 'car.data', 'cmc.data', 'CNAE-9.data', 'connect-4.data', 'covtype.data', 'dermatology.data', 'ecoli.data', 'fertility_Diagnosis.txt', 'flag.data', 'glass.data', 'hepatitis.data', 'house-votes-84.data', 'ionosphere.data', 'iris.data', 'krkopt.data', 'lung-cancer.data', 'mammographic_masses.data', 'pop_failures.dat', 'tictactoe.data', 'trains.data', 'transfusion.data', 'wine.data', 'zoo.data']

meta_data_lines.append("name,#numerical_features,#categorical_features,#samples")
for name in names:
	print("Dataset: %s\nNumerical features:%d\nCategorical features:%d" % (name, len(numerical_features[name]), len(categorical_features[name])))

	path = os.path.join(base_dir, name)
	

	if len(numerical_features[name]) > 0:
		# read numerical features

		numerical = np.genfromtxt(path, dtype=np.float, usecols=numerical_features[name],delimiter=delimiters[name],skip_header=header[name])

		# if there's only one collumn, transform shape=(n,) in shape=(n,1)
		if len(numerical_features[name]) == 1:
			numerical = numerical.reshape( (numerical.shape[0], 1))

		# replace missing values to mean value
		if name in missing:
			imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
			numerical = imp.fit_transform(numerical)			

	if len(categorical_features[name]) > 0:
		# read categorical features
		categorical = np.genfromtxt(path, dtype=np.unicode, usecols=categorical_features[name],delimiter=delimiters[name],skip_header=header[name])

		# replace missing values to most frequent value
		if name in missing:
			parseCategoricalMissingValues(categorical, missing_values=missing[name])

		# if there's only one collumn, transform shape=(n,) in shape=(n,1)
		if len(categorical_features[name]) == 1:
			categorical = categorical.reshape( (categorical.shape[0], 1))

		# transform categorical features in sequential integer categorical features, for each collumn
		numbered_categories = np.zeros(np.shape(categorical))
		m = np.shape(categorical)[1]
		for j in range(m):
			le = preprocessing.LabelEncoder()
			numbered_categories[:,j] = le.fit_transform(categorical[:,j])

		# one hot categorical encoding
		ohe = preprocessing.OneHotEncoder()
		normalized_categorical = ohe.fit_transform(numbered_categories).toarray()

	# join features
	if len(numerical_features[name]) > 0 and len(categorical_features[name]) > 0:
		features = np.concatenate( (numerical, normalized_categorical), axis=1)
	elif len(numerical_features[name]) == 0 and len(categorical_features[name]) > 0:
		features = normalized_categorical
	elif len(numerical_features[name]) > 0 and len(categorical_features[name]) == 0:
		features = numerical

	# normalize features
	min_max_scaler = preprocessing.MinMaxScaler((-1,1))
	features = min_max_scaler.fit_transform(features)
	normalized_features[name] = features
	np.savetxt(os.path.join(out_dir, "X_" + name + ".csv"), features, delimiter=",")

	# read classes
	classified = np.genfromtxt(path, dtype=np.unicode, usecols=classes[name],delimiter=delimiters[name],skip_header=header[name])

	# transform categorical classes in numeric classes
	le = preprocessing.LabelEncoder()
	numbered_classes = le.fit_transform(classified)
	
	# classes didn't need to be transposed
	normalized_classes[name] = numbered_classes.reshape( (numbered_classes.shape[0]) )
	np.savetxt(os.path.join(out_dir, "Y_" + name + ".csv"), normalized_classes[name], delimiter=",")

	print("Samples: %d\n" % len(normalized_classes[name]))
	meta_data_lines.append("%s,%d,%d,%d" % (name, len(numerical_features[name]), len(categorical_features[name]),len(normalized_classes[name])))

print("Processed datasets: %d" % len(names))
with open(meta_data_path, "w") as meta_data_file:
	meta_data_file.write("\n".join(meta_data_lines))
	meta_data_file.close()

pickle.dump(normalized_features, open(x_output_path, "wb"))
pickle.dump(normalized_classes, open(y_output_path, "wb"))
