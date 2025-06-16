import GeneticPlayerStuff.Helper_Functions as hf
import numpy as np
import os
import json
import datetime

from GeneticPlayerStuff.Genetic_Agent import GeneticPlayer
from pypokerengine.api.game import setup_config, start_poker
from concurrent.futures import ThreadPoolExecutor

POPULATION_SIZE = 7
EPOCHES = 200
MUTATION_RATE = 0.15
MAX_GAMES_PLAYED = 3
MAX_ROUNDS = 3
INITIAL_STACK = 100000
SMALL_BLIND = 100


class Population(object):
    def __init__(self, size, log_dir='../Logs/Epoches'):
        self.log_dir = log_dir
        self.pops = []
        self.size = size
        for _ in range(size):
            prob = hf.normalize(hf.get_def_probability() * (1 + np.random.uniform(-0.1, 0.1, size=(5, 3))))
            self.pops.append(GeneticPlayer(prob, np.random.uniform(0.1, 2)))

    def birth(self):
        # This method is called in each epoch.
        # It should then call compute_gen_fitness
        # Setup the generation of new players

        fitness_scores = self.compute_gen_fitness()
        self.log_summary(fitness_scores)
        fitness_scores = [i/sum(fitness_scores) for i in fitness_scores]

        new_gen = []
        for _ in range(self.size - 2):
            parent1, parent2 = self.select_parents(fitness_scores)
            child = self.crossover(parent1, parent2)
            child.mutate(MUTATION_RATE)
            new_gen.append(child)

        pop_fitness = list(zip(self.pops, fitness_scores))
        pop_fitness.sort(key=lambda x: x[1], reverse=True)
        new_gen.extend([x[0] for x in pop_fitness[:2]])

        self.pops = new_gen

    def compute_gen_fitness(self):
        # Prepare the game (call play_game) and compute fitness scores (chips left)
        total_fitness = [0] * self.size

        for i in range(MAX_GAMES_PLAYED):
            table = [(self.pops[i], i) for i in range(self.size)]

            fitness_scores = self.play_game(table)
            print(f"[LOG] Game {i + 1}/{MAX_GAMES_PLAYED} finished. Scores: {fitness_scores}")
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

    def select_parents(self, fitness_scores):
        # Modify the selection so that the best fitness scores are more likely to be selected
        total_fitness = sum(fitness_scores)
        probabilities = [f / total_fitness for f in fitness_scores]
        return np.random.choice(self.pops, size=2, p=probabilities)

    def crossover(self, parent1, parent2):
        child_prob = hf.add_lists([parent1.actions_prob, parent2.actions_prob])
        child_prob = [x / 2 for x in child_prob]
        child_prob = hf.normalize(child_prob)
        child_agg = (parent1.aggresion + parent2.aggresion) / 2
        return GeneticPlayer(child_prob, child_agg)

    def log_summary(self, fitness_scores, top=3):
        # Sort by fitness descending
        indexed_scores = list(enumerate(fitness_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        summary = {
            "top_agents": []
        }

        for i in range(min(top, len(indexed_scores))):
            idx, score = indexed_scores[i]
            player = self.pops[idx]
            summary["top_agents"].append({
                "index": idx,
                "fitness": score,
                "aggression": player.aggresion,
                "default_prob": np.array(player.actions_prob).tolist()
            })

        summary["max_fitness"] = float(max(fitness_scores))
        summary["mean_fitness"] = float(np.mean(fitness_scores))
        summary["min_fitness"] = float(min(fitness_scores))

        os.makedirs(self.log_dir, exist_ok=True)
        filename = os.path.join(self.log_dir, f"epoch_" + datetime.datetime.now().strftime("%Y-%m-%d %H-%M") + ".json")
        try:
            with open(filename, "w") as f:
                json.dump(summary, f, indent=4)
        except:
            print("[ERROR] Failed to save epoch summary.")
            print(summary)
            return

        print(f"[LOG] Saved epoch summary to {filename}")



if __name__ == "__main__":
    population = Population(POPULATION_SIZE)

    for epoch in range(EPOCHES):
        print(f"Epoch {epoch + 1}/{EPOCHES}")
        population.birth();