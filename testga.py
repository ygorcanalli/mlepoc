import numpy as np
from random import random, randint
from sklearn import cross_validation
from sklearn import svm
from sklearn import datasets
from bitstring import BitArray

max_gen 	= 1
pop_size 	= 3
p_cross 	= 0.9
p_mutation 	= 0.05
mutation_f	= 0.1

def main():
	
	gen = 1
	
	#inicializar população
	population = initializePopulation()
	print(population)

	#criterio de parada
	while True:

		fitness = evaluatePopulation(population)
		#newPopulation = selectNextPopulation(population, fitness)
		#crossover(newPopulation)
		mutation(population)
	
		#population = newPopulation
	
		gen += 1
		
		if gen >= max_gen:
		
			break

	print(population)
	
def initializePopulation():
	
	population = np.ones((pop_size, 2))
	
	for i in range(pop_size):
		population[i] = (random(), random())
	
	return population

def evaluatePopulation(population):
	
	#utilizando iris para simular o svm aplicado a um dataset qualquer
	iris = datasets.load_iris()
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

	fitness = np.zeros((pop_size))
	
	for i in range(pop_size):
		
		clf = svm.SVC(C=population[i][0], gamma=population[i][1]).fit(X_train, y_train)
		fitness[i] = clf.score(X_test, y_test)
		
	return fitness
	
#def selectNextPopulation():

#def crossover(population):	

def mutation(population):

	for i in range(pop_size):
		if p_mutation > random():
			k = randint(0,1)
			#adiciona fator
			if(k==0):
				j = randint(0,population[i].shape[0]-1)
				population[i][j] *= 1 + mutation_f
				
			#diminui fator
			else:
				j = randint(0,population[i].shape[0]-1)
				population[i][j] *= 1 - mutation_f					
	
main()
