# HetNet-PPO


This repository is based on: https://github.com/zoeyuchao/mappo


## Installation
Install Conda environment: 
```shell
conda env create -f environment.yml
conda activate hetnet_ppo
pip install -r requirements.txt
```

Please refer to the [PyTorch](https://pytorch.org/get-started/locally/) and the [Deep Graph Library](https://www.dgl.ai/pages/start.html) websites for installing the PyTorch and DGL libraries. As an example, in the following we install PyTorch and DGL to work with `CUDA==11.8`. 

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

Install the `onpolicy` package: 

```shell
pip install -e . 
```

### Running

Experiments for training HetNet-PPO can be run from the `onpolicy/scripts/train/` directory. The following are examples of running experiments for the Predator-Prey, Predator-Capture-Prey, and FireCommander environments.


#### Running Predator-Prey:

```shell
python train_predator.py --algorithm_name hetgat_mappo --ppo_epoch 5 --entropy_coef 0.01 --use_recurrent_policy --use_LSTM --hidden_size 128 --env_name PredatorCapture --n_types 2 --num_P 3 --num_A 0 --dim 5 --vision 0 --episode_limit 80 --episode_length 500 --num_env_steps 24000000 --seed 2 --experiment_name hetnet_ppo_pp --tensor_obs
```

#### Running Predator-Capture-Prey:
```shell
python train_predator.py --algorithm_name hetgat_mappo --ppo_epoch 5 --entropy_coef 0.01 --use_recurrent_policy --use_LSTM --hidden_size 128 --env_name PredatorCapture --n_types 2 --num_P 2 --num_A 1 --dim 5 --vision 0 --episode_limit 80 --episode_length 500 --num_env_steps 24000000 --seed 2 --experiment_name hetnet_ppo_pcp --tensor_obs
```


#### Running FireCommander:
```shell
python train_fire_commander.py --algorithm_name hetgat_mappo --ppo_epoch 5 --entropy_coef 0.01 --use_recurrent_policy --use_LSTM --hidden_size 128 --env_name FireCommander --n_types 2 --num_P 2 --num_A 1 --dim 5 --vision 1 --episode_limit 300 --num_env_steps 24000000 --seed 2 --experiment_name hetnet_ppo_5x5_2p1a_fc --tensor_obs --nfires 1 --seed 2
```



**Note:** 
- When the number of Action agents (--num_A) is set to 0, the Predator-Capture-Prey environment defaults to Predator-Prey.
- For full list of parameters, default values, and descriptions, please refer to `onpolicy/config.py` and the `parse_args()` function in `onpolicy/scripts/train/train_predator.py` and `onpolicy/scripts/train/train_fire_commander.py`.

