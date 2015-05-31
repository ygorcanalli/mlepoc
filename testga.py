import numpy as np
from random import random, randint, uniform
from sklearn import cross_validation
from sklearn import svm
from sklearn import datasets
from bitstring import BitArray

max_gen 	= 20
pop_size 	= 10
p_mutation 	= 0.05
mutation_f	= 0.1

def main():
	
	gen = 1
	
	#inicializar populacao
	population = initializePopulation()
	print(population)

	#criterio de parada
	while True:

		fitness = evaluatePopulation(population)
		parents = selectNextPopulation(population, fitness)
		newpopulation = crossover(parents)
		mutation(newpopulation)
	
		population = newpopulation
	
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
	
def selectNextPopulation(population, fitness):
	
	ranked = rank(fitness)
	sumfit = 0
	
	for i in range(pop_size):
		sumfit += ranked[i]
				
	parents = np.ones((pop_size, 2))
	
	for i in range(pop_size):
		k = uniform(0.0, sumfit)
		aux = 0.0
		for j in range(pop_size):
			aux += ranked[j]
			if k < aux:
				parents[i] = population[j]
				break
			
	return parents
	
def rank(fitness):
	
	newfitness = sorted(fitness)
	rankedfitness = np.zeros((pop_size))
	
	for i in range(pop_size):
		rankedfitness[i] = 100 * (newfitness[0] + (newfitness[-1] - newfitness[0]) * (fitness[i] - 1) / (pop_size - 1))
		
	return rankedfitness

def crossover(parents):
	newpopulation = np.zeros((pop_size, 2))
	k = 0
	
	for i in range(int(pop_size/2)):
		
		a11, a12 = parents[k]
		a21, a22 = parents[k+1]
		
		c11 = uniform(a11, a21)
		c12 = uniform(a12, a22)
		c21 = uniform(a11, a21)
		c22 = uniform(a12, a22)
		
		newpopulation[k] = [c11, c21]
		newpopulation[k+1] = [c12, c22]
		
		k += 2
		
	return newpopulation
		
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
