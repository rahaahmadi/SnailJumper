import copy
import random
import numpy as np
import math
from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        sorted_players = sorted(players, key=lambda player: player.fitness, reverse=True)
        best_fitness = sorted_players[0].fitness
        worst_fitness = sorted_players[len(sorted_players) - 1].fitness
        fitnesses = [player.fitness for player in players]
        mean_fitness = sum(fitnesses) / len(fitnesses)
        with open('learning_curve.txt', 'a') as f:
            f.write(f'{best_fitness} {worst_fitness} {mean_fitness} \n')

        method = 'rw'

        if method == 'rw':
            return self.roulette_wheel(players, num_players)
        elif method == 'sus':
            return self.sus(players, num_players)
        elif method == 'q':
            return self.q_tournament(players, num_players)
        else:
            return sorted_players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            method = 'q'
            if method == 'rw':
                prev_players = self.roulette_wheel(prev_players, num_players)
            elif method == 'sus':
                prev_players = self.sus(prev_players, num_players)
            elif method == 'q':
                prev_players = self.q_tournament(prev_players, num_players)

            new_players = self.cross_over(prev_players, num_players)
            new_players = [self.mutate(player) for player in new_players]
            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player


    def q_tournament(self, players, num_players):
        next_generation = []
        q = 10
        for i in range(num_players):
            candidates = random.sample(players, q)
            next_generation.append(max(candidates, key=lambda x: x.fitness))
        return next_generation

    def roulette_wheel(self, players, num_players):
        next_generation = []
        total_fitness = sum([p.fitness for p in players])
        probs = [p.fitness / total_fitness for p in players]
        next_generation = np.random.choice(players, size=num_players, p=probs)
        return list(next_generation)
    
    def sus(self, players, num_players):
        total_fitness = np.sum([p.fitness for p in players])
        point_distance = total_fitness / num_players
        start_point = np.random.uniform(0, point_distance)
        points = [start_point + i * point_distance for i in range(num_players)]
   
        next_generation = []
        for point in points:
            i = 0
            f = 0
            while f < point:
                f += players[i].fitness
                i += 1
            next_generation.append(players[i - 1])
        return next_generation

    
    def cross_over(self, players, num_players):
        children = []
        index = 0
        for i in range(math.floor(num_players / 2.0)):
            cross_over_prob = np.random.uniform(0, 1)
            p = 0.8
            if cross_over_prob >= p:
                children.append(players[index])
                children.append(players[index + 1])
                index += 2
                continue
            parent1 = players[index]
            parent2 = players[index + 1]
            child1 = self.clone_player(parent1)
            child2 = self.clone_player(parent2)

            d0 = math.floor(parent1.nn.layer_sizes[0] / 2)
            d1 = math.floor(parent1.nn.layer_sizes[1] / 2)
            d2 = math.floor(parent1.nn.layer_sizes[2] / 2)
            child1.nn.w1 = np.concatenate((parent1.nn.w1[:d1], parent2.nn.w1[d1:]), axis=0)
            child1.nn.b1 = np.concatenate((parent1.nn.b1[:d1], parent2.nn.b1[d1:]), axis=0)
            child1.nn.w2 = np.concatenate((parent1.nn.w2[:d2], parent2.nn.w2[d2:]), axis=0)
            child1.nn.b2 = np.concatenate((parent1.nn.b2[:d2], parent2.nn.b2[d2:]), axis=0)
            children.append(child1)
            child2.nn.w1 = np.concatenate((parent2.nn.w1[:d1], parent1.nn.w1[d1:]), axis=0)
            child2.nn.b1 = np.concatenate((parent2.nn.b1[:d1], parent1.nn.b1[d1:]), axis=0)
            child2.nn.w2 = np.concatenate((parent2.nn.w2[:d2], parent1.nn.w2[d2:]), axis=0)
            child2.nn.b2 = np.concatenate((parent2.nn.b2[:d2], parent1.nn.b2[d2:]), axis=0)
            children.append(child2)
            index += 2
        if len(children) < num_players:
            children.append(players[0])

        return children

    def mutate(self, child):
        mutation_probability = 0.3
        layer_sizes = child.nn.layer_sizes
        if random.uniform(0, 1) < mutation_probability:
            child.nn.w1 += np.random.randn(layer_sizes[1], layer_sizes[0])
        if random.uniform(0, 1) < mutation_probability:
            child.nn.w2 += np.random.randn(layer_sizes[2], layer_sizes[1])
        if random.uniform(0, 1) < mutation_probability:
            child.nn.b1 += np.random.randn(layer_sizes[1], 1)
        if random.uniform(0, 1) < mutation_probability:
            child.nn.b2 += np.random.randn(layer_sizes[2], 1)
        return child

