python -c "
from metadrive.envs.metadrive_env import MetaDriveEnv
env = MetaDriveEnv(dict(use_render=False, num_scenarios=1, map='CCC'))
obs, info = env.reset()

# Get current map and lanes
current_map = env.current_map
print(f'=== MAP ===')
print(f'Map type: {type(current_map).__name__}')
print(f'Road network: {type(current_map.road_network).__name__}')

# Inspect road network graph
rn = current_map.road_network
print(f'Graph keys: {list(rn.graph.keys())[:10]}')

# Try to get lane info
for road_id, lanes in list(rn.graph.items())[:3]:
    print(f'\nRoad: {road_id}')
    for to_road, lane_list in lanes.items():
        print(f'  -> {to_road}: {len(lane_list)} lanes')
        for i, lane in enumerate(lane_list):
            print(f'    Lane {i}: {type(lane).__name__}')
            # Try to get centerline points
            if hasattr(lane, 'position'):
                p0 = lane.position(0, 0)
                p1 = lane.position(lane.length, 0)
                print(f'      Start: {p0}, End: {p1}, Length: {lane.length:.1f}m')
            if hasattr(lane, 'width'):
                print(f'      Width: {lane.width}m')

env.close()
"
