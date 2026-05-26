# Running Evaluation

`hetnet_ppo_5x5_2p1a_vision_1_fire_commander_tensorized.sh` is an example evaluation script for a trained HetGAT-MAPPO model on the Fire Commander task.

## Key Arguments

| Argument | Description |
|---|---|
| `--algorithm_name` | Algorithm to evaluate (`hetgat_mappo`) |
| `--experiment_name` | Name for the run (used for logging) |
| `--use_eval` | Enables evaluation mode |
| `--eval_string` | Path to save evaluation results |
| `--eval_config` | `.pkl` file with initial conditions for eval episodes |
| `--model_dir` | Path to the saved model checkpoint |

## Output

After evaluation, `print_plot_eval.py` aggregates all `.pt` result files in the eval directory, printing and saving mean steps-to-completion and reward (with standard error) across seeds to `summary_eval.json`.