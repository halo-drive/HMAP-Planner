python -c "
from metadrive.envs.metadrive_env import MetaDriveEnv
env = MetaDriveEnv(dict(use_render=False, num_scenarios=1, traffic_density=0.3))
obs, info = env.reset()

navi = env.agent.navigation
print('=== NAVIGATION ===')
print(f'Type: {type(navi).__name__}')

# Check checkpoints
if hasattr(navi, 'checkpoints'):
    print(f'Checkpoints: {navi.checkpoints}')
if hasattr(navi, 'current_ref_lanes'):
    print(f'Current ref lanes: {len(navi.current_ref_lanes)}')
    for i, lane in enumerate(navi.current_ref_lanes[:3]):
        print(f'  Lane {i}: {type(lane).__name__}, length={lane.length:.1f}m')

# Navigation attributes
attrs = [a for a in dir(navi) if not a.startswith('_') and not callable(getattr(navi, a, None))]
for a in sorted(attrs):
    try:
        v = getattr(navi, a)
        print(f'  {a}: {v}')
    except:
        pass

env.close()
"
