import os
import json

for filename in sorted(os.listdir("../uci/")):
	
	if os.path.isfile(os.path.join("../uci", filename)):
		lines = []
		with open(os.path.join("../uci", filename), 'r') as f:
			lines = f.read().split('\n')
		params = json.loads(lines[0].replace("\'", "\""))
		test_accuracy = lines[1].split(': ')[1]
		training_accuracy = lines[2].split(': ')[1]
		original_name = filename.replace('optimal_', '')
		print("%s,%s,%s,%s,%s" % (original_name, params["gamma"], params["C"], test_accuracy, training_accuracy))
		