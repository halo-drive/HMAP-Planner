python -c "
from metadrive.envs.metadrive_env import MetaDriveEnv
env = MetaDriveEnv(dict(use_render=False, num_scenarios=1, traffic_density=0.3))
obs, info = env.reset()

agent = env.agent
print('=== EGO VEHICLE ===')
print(f'Position:      {agent.position}')
print(f'Heading theta: {agent.heading_theta}')
print(f'Speed (m/s):   {agent.speed / 3.6}')  # MetaDrive speed might be km/h
print(f'Velocity:      {agent.velocity}')
print(f'Type:          {type(agent.velocity)}')

# Check what attributes the agent has
print('\n=== AGENT ATTRIBUTES ===')
attrs = [a for a in dir(agent) if not a.startswith('_')]
for a in sorted(attrs):
    try:
        v = getattr(agent, a)
        if not callable(v):
            print(f'  {a}: {type(v).__name__} = {v}')
    except:
        pass

env.close()
"
