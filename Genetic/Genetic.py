import random
from numpy import vectorize

class GAlgo():
    
    def __init__(self,hash_map, start, steps = 2, crossover_prob = 0.15, 
                 mutation_prob = 0.15, population_size = 5, iterates = 100):
        self.crossover_prob=crossover_prob
        self.mutation_prob=mutation_prob
        self.population_size=population_size
        self.hash_map = hash_map
        self.steps = steps
        self.iterates = iterates
        self.start = start
        self.cities = [k for k in self.hash_map.keys()] 
        self.cities.remove(start)
        self.genes = []
        self.epsilon = 1 - 1 / self.iterates        
        self.generate_G = vectorize(self.generate_G)
        self.evaluate_fit = vectorize(self.evaluate_fit)
        self.evolve = vectorize(self.evolve)
        self.prune_genes = vectorize(self.prune_genes)
        self.converge = vectorize(self.converge)
        
        # Generating first population
        self.generate_G()

    # Gene generating function        
    def generate_G(self):
        for i in range(self.population_size):
            gene = [self.start]
            other_c = [k for k in self.cities]
            while len(gene) < len(self.cities) + 1:
                city = random.choice(other_c)
                gene.append(city)
                del other_c[other_c.index(city)]
            gene.append(self.start)
            self.genes.append(gene)
        return self.genes
    
    # Fitness function
    def evaluate_fit(self):
        fit_point = []
        for gene in self.genes:
            self.road = []
            total_distance = 0
            for idx in range(1,len(gene)):
                city_b = gene[idx]
                city_a = gene[idx-1]
                try:
                    dist = self.hash_map[city_a][city_b]
                except:
                    dist = self.hash_map[city_b][city_a]
                total_distance += int(dist)
                self.road.append(city_a)
            fitness = 1 / total_distance
            fit_point.append(fitness)
        return fit_point
    
    
    def evolve(self):
        index_map = {i:'' for i in range(1,len(self.cities)-1)}
        indices = [i for i in range(1,len(self.cities)-1)]
        to_visit = [c for c in self.cities]
        cross = (1 - self.epsilon) * self.crossover_prob
        mutate = self.epsilon * self.mutation_prob 
        crossed_count = int(cross * len(self.cities)-1)
        mutated_count = int((mutate * len(self.cities)-1)/2)
        for idx in range(len(self.genes)-1):
            gene = self.genes[idx]
            for i in range(crossed_count):
                try:
                    gene_index = random.choice(indices)
                    sample = gene[gene_index]
                    if sample in to_visit:
                        index_map[gene_index] = sample
                        loc = indices.index(gene_index)
                        del indices[loc]
                        loc = to_visit.index(sample)
                        del to_visit[loc]
                    else:
                        continue
                except:
                    pass
        last_gene = self.genes[-1]
        remaining_cities = [c for c in last_gene if c in to_visit]
        for k,v in index_map.items():
            if v != '':
                continue
            else:
                city = remaining_cities.pop(0)
                index_map[k] = city
        new_gene = [index_map[i] for i in range(1,len(self.cities)-1)]
        new_gene.insert(0,self.start)
        new_gene.append(self.start)
        for i in range(mutated_count):
            choices = [c for c in new_gene if c != self.start]
            city_a = random.choice(choices)
            city_b = random.choice(choices)
            index_a = new_gene.index(city_a)
            index_b = new_gene.index(city_b)
            new_gene[index_a] = city_b
            new_gene[index_b] = city_a
        self.genes.append(new_gene)
                
    def prune_genes(self):       
        for i in range(self.steps):
            self.evolve()
        fit_point = self.evaluate_fit()
        for i in range(self.steps):
            worst_gene_index = fit_point.index(min(fit_point))
            del self.genes[worst_gene_index]
            del fit_point[worst_gene_index]
        return max(fit_point),self.genes[fit_point.index(max(fit_point))]
    
    def converge(self):
        for i in range(self.iterates):
            values = self.prune_genes()
            current_score = values[0]
            current_best_gene = values[1]
            self.epsilon -= 1/self.iterates
            if i % 100 == 0:
                print(int(1/current_score), "miles")
        return current_best_gene, int(1/current_score)