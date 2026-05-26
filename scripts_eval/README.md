# Running Evaluation

`hetnet_a2c_fc_5x5_2p1a.sh` is an example evaluation script for a trained HetNet-A2C model on the Fire Commander task.

## Key Arguments

| Argument | Description |
|---|---|
| `--experiment_name` | Name for the run (used for logging) |
| `--save_dir` | Where to save experiment logs |
| `--eval` | Enables evaluation mode |
| `--eval_string` | Path to save evaluation results |
| `--eval_config` | `.pkl` file with initial conditions for eval episodes |
| `--load` | Path to the saved model checkpoint |

## Output

After evaluation, `print_plot_eval.py` aggregates all `.pt` result files in the eval directory, printing and saving mean steps-to-completion and reward (with standard error) across seeds to `summary_eval.json`.