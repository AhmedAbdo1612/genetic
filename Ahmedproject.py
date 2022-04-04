from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd 
from matplotlib import pyplot as plt
import random as rd 
import numpy as np 

cancer_data = load_breast_cancer()
features = cancer_data.data
labels = cancer_data.target
population_size = 20
number_of_generations = 500
initial_population = []
for i in range(population_size):
    ch = []
    for i in range(31):
        ch.append(rd.uniform(-5000, 5000))
    initial_population.append(ch)

def fitness(parm):
    error_list = []
    for i in range(len(features)):
        value = parm[0]
        for j in range(30):
            p = parm[j+1]*features[i][j]
            value+= p

        error = abs(labels[i] - value)
        error_list.append(error)
    fitness_vlaue =  (sum(error_list))

    return fitness_vlaue
ch_and_fitness = []

def meating(ch1,ch2):
    prob_mutation1 = rd.uniform(0, 1)
    prob_mutation2 = rd.uniform(0, 1)

    ch1_2 = ch1[0:int(len(ch1)/2)]+ ch2[int(len(ch2)/2):len(ch2)]
    ch2_1 = ch2[0:int(len(ch2)/2)]+ ch1[int(len(ch1)/2):len(ch1)]
    if prob_mutation1 >=0.5:
        index = rd.randint(0, len(ch1)-1)
        ch1_2[index] = ch1_2[index]*3/4
    if prob_mutation2 >=0.5:
        index = rd.randint(0, len(ch1)-1)
        ch2_1[index] = ch2_1[index]*3/4
    return ch1_2, ch2_1
    

for i in range(population_size):
    ch = initial_population[i]
    ch_fitness = fitness(ch)
    ch_fit = (ch_fitness,ch)
    ch_and_fitness.append(ch_fit)

ch_and_fitness =sorted(ch_and_fitness)

def create_new_generation(generation):
    offsprings = []
    for i in range(5):
        j= i+1
        while j <=4:
            ch1, ch2 = meating(generation[i][1], generation[j][1])
            ch1_fitness = fitness(ch1)
            offsprings.append((ch1_fitness,ch1))
            ch2_fitness = fitness(ch2)
            offsprings.append((ch2_fitness,ch2))
            j+=1

    # i = 0

    # while i <= len(ch_and_fitness)-2:
    #     ch1, ch2 = meating(generation[i][1], generation[i+1][1])
    #     ch1_fitness = fitness(ch1)
    #     offsprings.append((ch1_fitness,ch1))
    #     ch2_fitness = fitness(ch2)
    #     offsprings.append((ch2_fitness,ch2))
        
    #     i+=2
    return offsprings

for i in range(number_of_generations):
    ch_and_fitness= create_new_generation(ch_and_fitness) 
    ch_and_fitness = sorted(ch_and_fitness) 

def accuracy(ch):
    predition = []
    for i in range(len(features)):
        parameters = ch[0]
        for j in range(30):
            p = ch[j+1]*features[i][j]
            parameters+= p
        predicted_label = 1 / (1+np.exp(-parameters))
        predition.append(predicted_label)
    return predition
predicted_labels = accuracy(ch_and_fitness[0][1])
score = 0
for j in range(len(features)):
    if predicted_labels[j] == labels[j]:
        score +=1
    

print("The accurcy is:",(score/560)*100)
