from gymnasium.envs.registration import register

register(
    id="Snake-v0",  # this is the name you'll use in gym.make()
    entry_point="snake_env:SnakeEnv"  # path: module_name:class_name
)