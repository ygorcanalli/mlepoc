import numpy as np
from random import random, randint, uniform
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn import datasets
from bitstring import BitArray

n_folds = 4

def run(X, y, theta, elite = 1, max_gen = 100, pop_size = 10, p_cross = 0.9, p_mutation = 0.05, mutation_f = 0.1, stop_criteria = 10):

	gen = 1

	bestIndividual = np.zeros((len(theta)))
	bestFit = 0
	counter = 0

	#inicializar populacao
	population = initializePopulation(theta)

	#criterio de parada
	while True:
		
		k, fitness = evaluatePopulation(X, y, pop_size, population)
		parents = selectNextPopulation(population, pop_size, fitness, elite, k)
		newpopulation = crossover(parents, pop_size, p_cross)
		mutation(newpopulation, pop_size, p_mutation, mutation_f)
	
		population = newpopulation
		
		cIndex = np.argmax(fitness)
	
		if(fitness[cIndex] > bestFit):
			bestFit = fitness[cIndex]
			bestIndividual = population[cIndex]
			counter = 0

		gen += 1
		counter += 1
		
		if (gen >= max_gen or counter > stop_criteria):
		
			break

	k, fitness = evaluatePopulation(X, y, pop_size, population)
	index = np.argmax(fitness)
	
	print("Num generations: %d" % counter)
	return population[index], fitness[index]
	
def initializePopulation(theta):
	
	population = theta
		
	return population

def evaluatePopulation(X, y, pop_size, population):

	fitness = np.zeros((pop_size))
	
	kf = KFold(len(X), n_folds = n_folds)
	
	for i in range(pop_size):
	
		kfitness = np.zeros((n_folds))
		j=0
	
		for train, test in kf:
		
			X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
		
			clf = svm.SVC(C=population[i][1], gamma=population[i][0]).fit(X_train, y_train)
			kfitness[j] = clf.score(X_test, y_test)
			j = j + 1
			
		fitness[i] = np.average(kfitness)
		
	return np.argmax(fitness), fitness
	
def selectNextPopulation(population, pop_size, fitness, elite, k):
	
	ranked = rank(fitness, pop_size)
	sumfit = np.sum(ranked)
		
	parents = np.ones((pop_size * 2, 2))
	
	if(elite):
		parents[0] = population[k]
		parents[1] = population[k]
	else:
		for i in range(2):
			value = uniform(0.0, sumfit)
			aux = 0.0
			for j in range(pop_size):
				aux += ranked[j]
				if value < aux:
					parents[i] = population[j]
					break
	
	for i in range(2, (pop_size)*2):
		value = uniform(0.0, sumfit)
		aux = 0.0
		for j in range(pop_size):
			aux += ranked[j]
			if value < aux:
				parents[i] = population[j]
				break
			
	return parents
	
def rank(fitness, pop_size):
	
	newfitness = sorted(fitness)
	rankedfitness = np.zeros((pop_size))
	
	for i in range(pop_size):
		rankedfitness[i] = 100 * (newfitness[0] + (newfitness[-1] - newfitness[0]) * (fitness[i] - 1) / (pop_size - 1))
		
	return rankedfitness

def crossover(parents, pop_size, p_cross):
	newpopulation = np.zeros((pop_size, 2))
	k = 0
	
	for i in range(pop_size):
		
		value = random()
		
		if value <= p_cross:
		
			a11, a12 = parents[k]
			a21, a22 = parents[k+1]
			
			c1 = uniform(a11, a21)
			c2 = uniform(a12, a22)
			
			newpopulation[i] = [c1, c2]

		else:
		
			ind = randint(k, k+1)
			newpopulation[i] = parents[ind]
			
		k += 2
		
	return newpopulation
		
def mutation(population, pop_size, p_mutation, mutation_f):

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
