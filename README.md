# HetNet-PPO Robot Demo Instructions

This repository contains the code for the HetNet-PPO robot demo and to be used after training
has been completed!

The folder and file of interest in this branch is the robotarium_python_simulator and si_go_to_point_gt.py file.


Once a PPO agent has been trained and saved, a trajectory can be generated using the trained agent. For a 
2 perception and 1 action agent scenario, at each timestep, the location of each agent must be saved and the location of 
each fire must be saved. Note as the number of fires can vary, None can be used as a standin when there is not a fire.


The si_go_to_point_gt.py file is a script that can be used to generate a trajectory for the robotarium simulator. The script takes
in the waypoints of each agent and fire locations at each timestep and utilizes a go to point controller to route agents.
The go to point controller utilizes control barrier functions to avoid collisions. Fires will be projected in real time through the robotarium's
graphic display.

This file can be tested by running in a terminal. Further setup instructions needed to run the file correctly can be 
found in the robotarium_python_simulator folder's README.


Once the file has been created with the above information, the file must be uploaded here: https://www.robotarium.gatech.edu/
alongside the image files FireLogo.png (which contains the fire image) and GTLogo.png (which contains a background 5x5 grid)
