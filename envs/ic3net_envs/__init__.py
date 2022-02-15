from gym.envs.registration import register

register(
    id='PredatorPrey-v0',
    entry_point='ic3net_envs.predator_prey_env:PredatorPreyEnv',
)

register(
    id='PredatorCapture-v0',
    entry_point='ic3net_envs.predator_capture_env:PredatorCaptureEnv',
)

register(
    id='FireCommander-v0',
    entry_point='ic3net_envs.fire_commander_env:FireCommanderEnv',
)

register(
    id='TrafficJunction-v0',
    entry_point='ic3net_envs.traffic_junction_env:TrafficJunctionEnv',
)
