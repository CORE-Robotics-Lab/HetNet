import json
import numpy as np
import pickle


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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


        # json_dump = json.dumps(initial_conditions, cls=NumpyEncoder)
        # with open(path, 'w') as f:
        #     json.dump(json_dump, f)

    # def load_initial_conditions(self, path):
    #     with open(path, 'r') as f:
    #         data = json.load(f)
    #         initial_conditions = json.loads(data)
    #
    #     return np.array(initial_conditions).reshape(100, 4, 2) # num locations, num agents, 2

eval = EvaluationConditions()

eval.generate_initial_conditions(size=5, num_agents=3, path='test_config/5x5_2p1a_initial_conditions.pkl')
eval.generate_initial_conditions(size=10, num_agents=5, path='test_config/10x10_3p2a_initial_conditions.pkl')
eval.generate_initial_conditions(size=20, num_agents=10, path='test_config/20x20_6p4a_initial_conditions.pkl')
