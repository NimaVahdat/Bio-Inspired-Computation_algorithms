import numpy as np

from SAC import AntColony

# Importing the data
data = open('Data1.txt', 'r')

distances = []
for i in range(23):
    line = data.readline()
    if i > 2:
        distances.append([int(x) for x in line.strip().split()])
for i in range(len(distances)):
    for j in range(len(distances)):
        if distances[i][j] == 0:
            distances[i][j] = np.inf
        
distances = np.array(distances[:])


ant_colony = AntColony(distances, 20, 20, 100, 0.95, alpha = 1, beta = 0)
shortest_path = ant_colony.run()
print ("\nbest:", shortest_path)