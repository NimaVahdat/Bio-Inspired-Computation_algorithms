import numpy as np
import random

class PSA():
    def __init__(self, function, partical_n, w, c1, c2, iteration):
        self.function = self.f1 if function == "f1" else self.f2
        self.bound = [-10, 10] if function == "f1" else [-100, 100]
        self.partical_n = partical_n
        self.particales = []
        self.GBest = [15, 15]
        self.PBest = []
        self.gen = 0
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v = [[0,0]] * self.partical_n
        self.iteration = iteration
        
        
    
    def f1(self, x, y):
        return -abs(np.sin(x) * np.cos(x) * \
                    np.exp(abs(1 - (np.sqrt(x**2 + y**2) / np.pi))))
    
    def f2(self, x, y):
        return 0.5 + (np.cos(np.sin(abs(x**2 - y**2))) ** 2 - 0.5) / \
            (1 + 0.001 * (x**2 + y**2))**2
       
    def init_pop(self):
        for i in range(self.partical_n):
            x = random.uniform(self.bound[0], self.bound[1])
            y = random.uniform(self.bound[0], self.bound[1])
            self.particales.append([x, y])
            self.PBest.append([x, y])
        return

    def fitness(self):
        for i in range(self.partical_n):
            if self.function(self.particales[i][0], self.particales[i][1]) < \
                self.function(self.PBest[i][0], self.PBest[i][1]):
                    self.PBest[i] = self.particales[i]
            
            if self.function(self.particales[i][0], self.particales[i][1]) < \
                self.function(self.GBest[0], self.GBest[1]):
                    self.GBest = self.particales[i]
        self.gen += 1
        if self.gen == 100:         
            print("Best till now:", self.function(self.GBest[0], self.GBest[1]),
                  "At point:", self.GBest)
            self.gen = 0
        return
    
    def velocity_update(self):
        for i in range(self.partical_n):
            self.v[i][0] = self.w * self.v[i][0] + self.c1 * random.uniform(0, 1) * (self.PBest[i][0] - self.particales[i][0])\
                + self.c2 * random.uniform(0, 1) * (self.GBest[0] - self.particales[i][0])
            
            self.v[i][1] = self.w * self.v[i][1] + self.c1 * random.uniform(0, 1) * (self.PBest[i][1] - self.particales[i][1])\
                + self.c2 * random.uniform(0, 1) * (self.GBest[1] - self.particales[i][1])
                
        return
    
    def position_update(self):
        self.velocity_update()
        for i in range(self.partical_n):
            self.particales[i][0] = self.particales[i][0] + self.v[i][0]
            self.particales[i][1] = self.particales[i][1] + self.v[i][1]
            
        return
    
    def run(self):
        self.init_pop()
        self.fitness()
        for i in range(self.iteration):
            self.position_update()
            self.fitness()
            
        return self.GBest, self.function(self.GBest[0], self.GBest[1])
    
    
n = PSA(function="f2", partical_n=200, w=0.1, c1=0.5, c2=1, iteration=1000)
point, score = n.run()

print("\nBest Score:", score)
print("At point:", point)