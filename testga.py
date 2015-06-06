import numpy as np
from random import random, randint, uniform
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn import datasets
from bitstring import BitArray

def run(X, y, theta, max_gen = 100, pop_size = 10, p_cross = 0.9, p_mutation = 0.05, mutation_f = 0.1):

	gen = 1

	#inicializar populacao
	population = initializePopulation(theta)
	
	#criterio de parada
	while True:
		
		fitness = evaluatePopulation(X, y, pop_size, population)
		parents = selectNextPopulation(population, pop_size, fitness)
		newpopulation = crossover(parents, pop_size, p_cross)
		mutation(newpopulation, pop_size, p_mutation, mutation_f)
	
		population = newpopulation
	
		gen += 1
		
		if gen >= max_gen:
		
			break

	print(population)
	
def initializePopulation(theta):
	
	population = theta
		
	return population

def evaluatePopulation(X, y, pop_size, population):

	fitness = np.zeros((pop_size))
	
	kf = KFold(len(X), n_folds = 10)
	
	for i in range(pop_size):
	
		kfitness = np.zeros((10))
		j=0
	
		for train, test in kf:
		
			X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
		
			clf = svm.SVC(C=population[i][0], gamma=population[i][1]).fit(X_train, y_train)
			kfitness[j] = clf.score(X_test, y_test)
			j = j + 1
			
		fitness[i] = np.average(kfitness)
		
	return fitness
	
def selectNextPopulation(population, pop_size, fitness):
	
	ranked = rank(fitness, pop_size)
	sumfit = np.sum(ranked)
		
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
	
def rank(fitness, pop_size):
	
	newfitness = sorted(fitness)
	rankedfitness = np.zeros((pop_size))
	
	for i in range(pop_size):
		rankedfitness[i] = 100 * (newfitness[0] + (newfitness[-1] - newfitness[0]) * (fitness[i] - 1) / (pop_size - 1))
		
	return rankedfitness

def crossover(parents, pop_size, p_cross):
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