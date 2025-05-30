import GeneticPlayerStuff.Helper_Functions as hf
import numpy as np

from GeneticPlayerStuff.Genetic_Agent import GeneticPlayer
from pypokerengine.api.game import setup_config, start_poker

POPULATION_SIZE = 5
EPOCHES = 20
MUTATION_RATE = 0.1
MAX_GAMES_PLAYED = 1
MAX_ROUNDS = 1
INITIAL_STACK = 200
SMALL_BLIND = 1


class Population(object):
    def __init__(self, size):
        self.pops = []
        self.size = size
        for _ in range(size):
            prob = hf.normalize(hf.get_def_probability() * (1 + np.random.uniform(-0.1, 0.1, size=(5, 3))))
            self.pops.append(GeneticPlayer(prob, np.random.uniform(0.1, 2)))

    def birth(self):
        # This method is called in each epoch. It should then call compute_gen_fitness
        # Setup the generation of new players

        fitness_scores = self.compute_gen_fitness()
        fitness_scores = [i/sum(fitness_scores) for i in fitness_scores]

        new_gen = []
        for _ in range(self.size):
            parent1, parent2 = self.select_parents(fitness_scores)
            child = self.crossover(parent1, parent2)
            child.mutate(MUTATION_RATE)
            new_gen.append(child)

        self.pops = new_gen

    def select_parents(self, fitness_scores):
        # Modify the selection so that the best fitness scores are more likely to be selected
        total_fitness = sum(fitness_scores)
        probabilities = [f / total_fitness for f in fitness_scores]
        return np.random.choice(self.pops, size=2, p=probabilities)

    def crossover(self, parent1, parent2):
        child_prob = hf.add_lists([parent1.default_prob, parent2.default_prob])
        child_prob = [x / 2 for x in child_prob]
        child_prob = hf.normalize(child_prob)
        child_agg = (parent1.aggresion + parent2.aggresion) / 2
        return GeneticPlayer(child_prob, child_agg)

    def compute_gen_fitness(self):
        # Prepare the game (call play_game) and compute fitness scores (chips left)
        total_fitness = [0] * self.size

        for i in range(MAX_GAMES_PLAYED):
            table = [(self.pops[i], i) for i in range(self.size)]

            fitness_scores = self.play_game(table)

            total_fitness = hf.add_lists([total_fitness, fitness_scores])

        return total_fitness

    def play_game(self, players):

        # Players is a tuple of (GeneticPlayer, name)

        # Play a game of poker with the given players

        config = setup_config(MAX_ROUNDS, INITIAL_STACK, SMALL_BLIND)

        for player, name in players:
            config.register_player(name=name, algorithm=player)

        results = start_poker(config, verbose=0)

        round_fitness = [0] * self.size
        for player in results['players']:
            round_fitness[player['name']] = player['stack']

        return round_fitness


if __name__ == "__main__":
    population = Population(POPULATION_SIZE)

    for epoch in range(EPOCHES):
        print(f"Epoch {epoch + 1}/{20}")
        population.birth();