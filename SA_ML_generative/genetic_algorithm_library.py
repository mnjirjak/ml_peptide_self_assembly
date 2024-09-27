import numpy as np


class GeneticAlgorithm:
    class Peptide:
        def __init__(self, sequence):
            self.sequence = sequence
            self.fitness = 0

    def __init__(
        self,
        fitness_function,
        similarity_penalty,
        length_penalty,
        min_initial_peptide_length,
        max_initial_peptide_length,
        allowed_amino_acids,
        population_size,
        offspring_count,
        max_num_generations,
        tournament_size,
        mutation_probability
    ):
        self.fitness_function = fitness_function
        self.similarity_penalty = similarity_penalty
        self.length_penalty = length_penalty
        self.min_initial_peptide_length = min_initial_peptide_length
        self.max_initial_peptide_length = max_initial_peptide_length
        self.allowed_amino_acids = allowed_amino_acids
        self.population_size = population_size
        self.offspring_count = offspring_count
        self.max_num_generations = max_num_generations
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability

    def find_peptides(self):
        population = self.generate_random_population()
        generation = 1

        while True:
            if generation > self.max_num_generations:
                break

            print(f"Generation: {generation}/{self.max_num_generations}")

            self.evaluate_population(population)
            offspring = self.generate_offspring(population)

            population += offspring
            self.evaluate_population(population)

            population = self.next_generation(population)
            generation += 1

        return population

    def generate_random_population(self):
        population = []

        for _ in range(self.population_size):
            sequence = ""
            for _ in range(np.random.randint(self.min_initial_peptide_length, self.max_initial_peptide_length + 1)):
                sequence += self.allowed_amino_acids[np.random.randint(len(self.allowed_amino_acids))]
            
            population.append(self.Peptide(sequence))

        return population

    def evaluate_population(self, population):
        for peptide in population:
            peptide.fitness = \
                self.fitness_function(peptide.sequence) - \
                self.similarity_penalty(peptide.sequence, population) - \
                self.length_penalty(peptide.sequence)

    def generate_offspring(self, population):
        offspring = []

        for _ in range(self.offspring_count):
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            child = self.recombination(parent1, parent2)
            child = self.mutation(child)
            
            offspring.append(child)

        return offspring

    def tournament_selection(self, population):
        random_parent = population[np.random.randint(len(population))]
        
        for _ in range(self.tournament_size):
            i = np.random.randint(len(population))
            if population[i].fitness > random_parent.fitness:
                random_parent = population[i]
        
        return random_parent

    def recombination(self, parent1, parent2):
        p = np.random.rand()
        sequence = parent1.sequence[:int(len(parent1.sequence) * p)] + parent2.sequence[int(len(parent2.sequence) * p):]

        return self.Peptide(sequence)

    def mutation(self, child):
        sequence = list(child.sequence[:])

        if np.random.rand() < self.mutation_probability:
            r = np.random.rand()

            if 0 <= r < 0.25:
                sequence.insert(
                    np.random.randint(len(sequence) + 1),
                    self.allowed_amino_acids[
                        np.random.randint(len(self.allowed_amino_acids))
                    ]
                )
            elif 0.25 <= r < 0.5:
                if len(sequence) > 1:
                    first_position, second_position = np.random.randint(len(sequence), size=2)
                    sequence[first_position], sequence[second_position] = \
                        sequence[second_position], sequence[first_position]
            elif 0.5 <= r < 0.75:
                if len(sequence) > 1:
                    del sequence[np.random.randint(len(sequence))]
            if 0.75 <= r <= 1:
                sequence[np.random.randint(len(sequence))] = \
                    self.allowed_amino_acids[np.random.randint(len(self.allowed_amino_acids))]

        sequence = "".join(sequence)

        return self.Peptide(sequence)

    def next_generation(self, population):
        sorted_population = sorted(population, key=lambda peptide: peptide.fitness)
        return sorted_population[-self.population_size:]
