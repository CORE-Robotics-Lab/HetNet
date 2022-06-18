# HetNet
Public implementation of Heterogeneous Policy Networks (HetNet) from AAMAS'22

**Paper Title:** Learning Efficient Diverse Communication for Cooperative Heterogeneous Teaming

**Authors:** Esmaeil Seraj*, Zheyuan Wang*, Rohan Paleja*, Daniel Martin, Matthew Sklar, Anirudh Patel, Matthew Gombolay

**Paper Link:** https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf



## Installation

```
git clone https://github.com/CORE-Robotics-Lab/HetNet
cd HetNet/envs
python setup.py develop
```

We **note** this repo and several files maintained are pulled/modified from https://github.com/IC3Net/IC3Net.

## Sample Run Command for HetNet  
- Predator-Prey: ``python main.py --env_name predator_prey --nfriendly_P 3 --nfriendly_A 0 --nprocesses 1 --num_epochs 2000 --hid_size 128 --detach_gap 5 --lrate 0.0001 --dim 5 --batch_size 500 --max_steps 80 --hetgat --hetgat_a2c --seed 5``
- Predator-Capture: ``python main.py --env_name predator_capture --nfriendly_P 2 --nfriendly_A 1 --nprocesses 1 --num_epochs 2000 --hid_size 128 --detach_gap 5 --lrate 0.0001 --dim 5 --batch_size 500 --max_steps 80 --hetgat --hetgat_a2c --seed 5``
- Fire-Commander: ``python main.py --env_name fire_commander --nfriendly_P 2 --nfriendly_A 1 --nprocesses 1 --num_epochs 2000 --hid_size 128 --detach_gap 5 --lrate 0.0001 --dim 5 --batch_size 500 --max_steps 80 --hetgat --hetgat_a2c --seed 5``

## Sample Run Commands for Baselines
This is currently a work in progress. Baselines that are supported are MAGIC, IC3Net, CommNet, and TarMAC. We are doing some large refactoring. Please email us with any urgent concerns!


## Citation
If you use this work and/or this codebase in your research, we ask you to please cite the original AAMAS'22 paper as shown below:

```
@inproceedings{seraj2022learning,
  title={Learning efficient diverse communication for cooperative heterogeneous teaming},
  author={Seraj, Esmaeil and Wang, Zheyuan and Paleja, Rohan and Martin, Daniel and Sklar, Matthew and Patel, Anirudh and Gombolay, Matthew},
  booktitle={Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems},
  pages={1173--1182},
  year={2022}
}
```



## License
Code is available under MIT license.

## Appendix: The FireCommander Domain
A detailed description of the heterogeneous multi-agent domain that we created for our experiments, the FireCommander environment,can be accessed through the following links:

**FireCommander arXiv Paper:** FireCommander: An Interactive, Probabilistic Multi-agent Environment for Heterogeneous Robot Teams

**FireCommander Paper Link:** https://arxiv.org/abs/2011.00165

**FireCommander Codebase:** https://github.com/EsiSeraj/FireCommander2020


