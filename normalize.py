import numpy as np
from sklearn import preprocessing

float_collumns = [0,1,2,3]
categorical_collumns = [4]

floats = np.genfromtxt("uci/iris.data", dtype=np.float, usecols=float_collumns,delimiter=",")
categorical = np.genfromtxt("uci/iris.data", dtype=np.unicode, usecols=categorical_collumns,delimiter=",")

if len(float_collumns) == 1:
	floats = floats.reshape( (floats.shape[0], 1))

le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()
min_max_scaler = preprocessing.MinMaxScaler()

numbers_categorical = le.fit_transform(categorical)

if len(categorical_collumns) == 1:
	numbers_categorical = numbers_categorical.reshape( (numbers_categorical.shape[0], 1))

normalized_categorical = ohe.fit_transform(numbers_categorical).toarray()

dataset = np.concatenate( (floats, normalized_categorical), axis=1)
dataset = min_max_scaler.fit_transform(dataset)
print(dataset)