import torch
import numpy as np
import os
import sys



if __name__ == "__main__":

    args = sys.argv[1:]

    files = []
    for file in os.listdir(args[0]):
        if file.endswith(".pt"):
            print(os.path.join(args[0], file))
            files.append(os.path.join(args[0], file))

    steps = []
    rewards = []
    for each_file in files:
        seed_data = torch.load(each_file)
        steps.extend(seed_data['steps_taken'])
        rewards.extend(seed_data['reward'])

    print('Average performance over three seeds is :', np.mean(steps), ' with standard ERROR: ',
          np.std(steps) / np.sqrt(len(steps)))
    print('Average reward over three seeds is :', np.mean(rewards), ' with standard ERROR: ',
          np.std(rewards) / np.sqrt(len(rewards)))
