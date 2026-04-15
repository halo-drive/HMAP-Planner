python -c "
from metadrive.envs.metadrive_env import MetaDriveEnv
env = MetaDriveEnv(dict(use_render=False, num_scenarios=1, traffic_density=0.2))
obs, info = env.reset()
print(f'Obs shape: {obs.shape}')
print(f'Obs dtype: {obs.dtype}')
for i in range(10):
    obs, reward, terminated, truncated, info = env.step([0, 0.1])
print('MetaDrive running OK')
env.close()
"
