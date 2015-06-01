import pickle

probe_names = ['ecoli.data', 'flag.data', 'glass.data','wine.data', 'zoo.data']
x_input_path = "normalized_X.pkl"
y_input_path = "normalized_Y.pkl"
probe_prefix = "PROBE_"
X_probe = dict()
y_probe = dict()

# Loading datasets
with open(x_input_path, 'rb') as f:
    X_all = pickle.load(f)

with open(y_input_path, 'rb') as f:
    y_all = pickle.load(f)

# save probe dataset for testing the cassification code
for probe_name in probe_names:
	X_probe[probe_name] = X_all[probe_name]
	y_probe[probe_name] = y_all[probe_name]

pickle.dump(X_probe, open(probe_prefix + x_input_path, "wb"))
pickle.dump(y_probe, open(probe_prefix + y_input_path, "wb"))