python -c "
from metadrive.envs.metadrive_env import MetaDriveEnv
env = MetaDriveEnv(dict(use_render=False, num_scenarios=1, traffic_density=0.5))
obs, info = env.reset()

# Step a few times so traffic spawns
for _ in range(20):
    obs, reward, terminated, truncated, info = env.step([0, 0.5])

# Get all traffic vehicles
tm = env.engine.traffic_manager
print(f'=== TRAFFIC VEHICLES: {len(tm.spawned_objects)} ===')

for vid, vehicle in list(tm.spawned_objects.items())[:5]:
    print(f'\nVehicle: {vid}')
    print(f'  Position:  {vehicle.position}')
    print(f'  Heading:   {vehicle.heading_theta}')
    print(f'  Speed:     {vehicle.speed}')
    print(f'  Velocity:  {vehicle.velocity}')
    print(f'  Length:    {vehicle.LENGTH}')
    print(f'  Width:     {vehicle.WIDTH}')

env.close()
"
