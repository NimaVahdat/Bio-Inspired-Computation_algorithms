from MemTSP import MemAlgo

# Importing the data
data = open('Data1.txt', 'r')

distances = []
for i in range(23):
    line = data.readline()
    if i > 2:
        distances.append(line.strip().split())

def get_distance(start, stop):
    distance = distances[int(start)-1][int(stop)-1]
    return distance
    
cities = """
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
"""
cities = [c for c in cities.split('\n') if c != '']

edges = []
dist_dict = {c:{} for c in cities}
for idx_1 in range(0,len(cities)-1):
    for idx_2 in range(idx_1+1,len(cities)):
        city_a = cities[idx_1]
        city_b = cities[idx_2]
        dist = get_distance(city_a,city_b)
        dist_dict[city_a][city_b] = dist
        edges.append((city_a,city_b,dist))
        
if __name__ == "__main__":
    
    g = MemAlgo(hash_map = dist_dict, start = '1', mutation_prob = 0.25, crossover_prob = 0.25,
                  population_size = 30, steps = 15, iterates = 2000, mempop = 2)
    
    best, score = g.converge()

    print("Best solution: ", best)
    print("Best fitness: ", score)   