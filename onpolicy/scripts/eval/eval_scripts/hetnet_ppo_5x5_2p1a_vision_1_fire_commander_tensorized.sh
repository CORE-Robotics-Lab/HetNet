
python eval_fire_commander.py --algorithm_name hetgat_mappo --ppo_epoch 5 --entropy_coef 0.01 --use_recurrent_policy --use_LSTM --hidden_size 128 --env_name FireCommander --n_types 2 --num_P 2 --num_A 1 --dim 5 --vision 1 --episode_limit 300 --num_env_steps 10000000 --seed 2 --experiment_name evaluation --tensor_obs --use_eval --model_dir /path/to/saved/model/ --eval_string /path/to/eval/result/seed/ --eval_config 5x5_2p1a_initial_conditions.pkl --nfires 1 --reward_type 3


python print_plot_eval.py /path/to/eval/result/


