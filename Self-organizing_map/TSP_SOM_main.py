import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

class TSP_SOM():
    def __init__(self, cities, alpha):
        self.cities = cities
        self.n = len(cities)
        self.nodes = [0] * self.n
        self.r = (self.n - 1) // 2
        self.weights = np.random.randint(np.array(self.cities).min(), 
                        np.array(self.cities).max(), size = (self.n, 2))
        self.alpha = alpha
        self.nodes = []
        self.fit = 0
        
        self.d = [0] * self.n
    
    def D(self, city, i):
        cit = np.array([city] * self.n)
        d = np.sum(((self.weights - cit)**2), axis = 1)
        index = np.argmin(d)
        if self.d[i] == index:
            self.flag = False
        else:
            self.flag = True
            self.d[i] = index
        return index
     
    def update(self, index, city):
        self.weights[index] = np.dot((1 - self.alpha), self.weights[index]) + np.dot(self.alpha, city)
        for i in range(1, self.r):
            self.weights[(index + i) % self.n] = np.dot((1 - self.alpha), self.weights[(index + i) % self.n]) + np.dot(self.alpha, city) 
            self.weights[(index - i) % self.n] = np.dot((1 - self.alpha), self.weights[(index - i) % self.n]) + np.dot(self.alpha, city)
    
    def road(self):
        for i in range(self.n):
            wei = np.array([self.weights[i]] * self.n)
            d = np.sum(((self.cities - wei)**2), axis = 1)
            index = np.argmin(d)
            self.nodes.append(self.cities[index])
            self.cities = np.delete(self.cities, index, axis = 0)
            self.n -= 1
        self.nodes = np.array(self.nodes)
        return
    
    def fitness(self):
        for i in range(len(self.nodes)):
            self.fit += np.sqrt(np.sum((self.nodes[i] - self.nodes[i - 1]) ** 2))
        return self.fit
    
    def plot(self):
        pl = np.append(self.nodes, [self.nodes[0]], axis = 0)
        plt.plot(pl[:, 0], pl[:, 1])
        plt.scatter(pl[:, 0], pl[:, 1], s=40, edgecolor='k')
        plt.scatter(self.weights[:, 0], self.weights[:, 1], c='Red')
        plt.show()
    
    def find(self):
        self.flag = True
        itere = 1
        while self.flag:#for i in range(14):#
            for i in range(self.n):
                index = self.D(self.cities[i], i)
                self.update(index, self.cities[i])
            self.alpha = self.alpha * 0.951
            if itere % 3 == 0:
                self.r -= 1
            itere += 1
        self.road()
        print(self.nodes)
        print("Best Fitness:", self.fitness())
        self.plot()
        # print(itere)
            

flag = False
coords = []
with open("bayg29.tsp", "r") as f:
    for line in f.readlines():
        line = line.split()
        if line == ["EOF"]:
            flag = False
        if flag:
            line0 = [float(x.replace("\n", "")) for x in line]
            coords.append(line0[1:])
        if line == ["DISPLAY_DATA_SECTION"]:
            flag = True
TSP = TSP_SOM(coords, 0.82) 
TSP.find()
end = time.time()
print("Time:", end - start)  