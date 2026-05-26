import json
import numpy as np
import pickle

# scripts to generate initial conditions
class EvaluationConditions():
    def __init__(self):

        self.num_5x5_locations = 100
        self.num_10x10_locations = 100
        self.num_20x20_locations = 100

        self.num_target = 1

        self.use_replacement = False
        self.seed = 0
        np.random.seed(self.seed)

    def generate_initial_conditions(self, size, num_agents, path):
        initial_conditions = []
        for _ in range(self.num_5x5_locations):
            x_coords = np.random.choice(size, num_agents + self.num_target, replace=self.use_replacement)
            y_coords = np.random.choice(size, num_agents + self.num_target, replace=self.use_replacement)

            initial_condition = np.stack((x_coords, y_coords), axis=1)
            initial_conditions.append(initial_condition)

        with open(path, 'wb') as f:
            pickle.dump(initial_conditions, f)

eval = EvaluationConditions()

eval.generate_initial_conditions(size=5, num_agents=3, path='5x5_2p1a_initial_conditions.pkl')
eval.generate_initial_conditions(size=10, num_agents=5, path='10x10_3p2a_initial_conditions.pkl')
eval.generate_initial_conditions(size=20, num_agents=10, path='20x20_6p4a_initial_conditions.pkl')
