"""Extract a single clockwise reference loop around the entire track.

This creates ONE continuous path for autonomous driving:
- One complete lap (no branching, no multiple lanes)
- Clockwise direction
- Smooth trajectory suitable for Pure Pursuit / PID control
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from simple_track import SimpleTrackEnv, LANE_NUM, LANE_WIDTH


def extract_single_clockwise_loop(env):
    """Extract ONE clockwise loop following the outer boundary.
    
    Returns:
        Dict with keys: 'path_x', 'path_y', 's_distances' (cumulative distance)
    """
    from metadrive.component.pgblock.first_block import FirstPGBlock
    
    road_network = env.current_map.road_network
    
    path_x = []
    path_y = []
    s_distances = []
    
    # Start at NODE_2 and follow right-turning connections (clockwise)
    current_node = FirstPGBlock.NODE_2
    
    visited_nodes = set()
    max_iterations = len(road_network.graph) + 10  # Safety limit
    start_pos = None
    close_gap_minsq = float('inf')
    
    for iteration in range(max_iterations):
        if current_node not in road_network.graph:
            break
            
        # Find outgoing lane to next node
        found_next = False
        for to_node, lane_list in road_network.graph[current_node].items():
            if not lane_list:
                continue
                
            # Get first lane
            lane = lane_list[0]
            
            # Sample points along this lane centerline
            num_samples = max(15, int(lane.length))
            
            for i in range(num_samples + 1):
                s = (lane.length / num_samples) * i
                x, y = lane.position(s, 0)
                
                # Skip duplicate at connection point
                if len(path_x) > 0:
                    last_x, last_y = path_x[-1], path_y[-1]
                    if abs(x - last_x) < 0.01 and abs(y - last_y) < 0.01:
                        continue
                
                path_x.append(x)
                path_y.append(y)
                
                if len(path_x) == 1:
                    s_dist = 0.0
                    if start_pos is None:
                        start_pos = (x, y)
                else:
                    prev_dist = s_distances[-1]
                    dx = x - path_x[-2]
                    dy = y - path_y[-2]
                    s_dist = prev_dist + np.sqrt(dx**2 + dy**2)
                
                s_distances.append(s_dist)
            
            # Track progress
            print(f"Segment {len(path_x)//15:3d}: length={lane.length:.1f}m")
            
            found_next = True
            current_node = to_node
        
        if not found_next:
            break
        
        # Check if we're back near start
        if start_pos and len(path_x) > 100:
            last_x, last_y = path_x[-1], path_y[-1]
            dist_sq = (last_x - start_pos[0])**2 + (last_y - start_pos[1])**2
            if dist_sq < close_gap_minsq:
                close_gap_minsq = dist_sq
    
    print(f"\n✓ Built path: {len(path_x)} points")
    print(f"✓ Total length: {s_distances[-1]:.1f} m")
    
    return {
        'path_x': path_x,
        'path_y': path_y, 
        's_distances': s_distances
    }


def smooth_path(path_dict, sigma=2.0):
    """Apply Gaussian smoothing to reduce high-frequency noise."""
    x = np.array(path_dict['path_x'])
    y = np.array(path_dict['path_y'])
    
    x_smooth = gaussian_filter1d(x, sigma)
    y_smooth = gaussian_filter1d(y, sigma)
    
    return {
        'path_x': x_smooth.tolist(),
        'path_y': y_smooth.tolist(),
        's_distances': path_dict['s_distances']
    }


def downsample_path(path_dict, min_distance=2.0):
    """Downsample to maintain minimum distance between points."""
    x = np.array(path_dict['path_x'])
    y = np.array(path_dict['path_y'])
    s = np.array(path_dict['s_distances'])
    
    keep_idx = [0]
    
    for i in range(1, len(x)):
        if s[i] - s[keep_idx[-1]] >= min_distance:
            keep_idx.append(i)
    
    return {
        'path_x': x[keep_idx].tolist(),
        'path_y': y[keep_idx].tolist(),
        's_distances': s[keep_idx].tolist()
    }


def visualize_with_vehicle_markers(path_dict, title="Reference Path with Vehicle Positions"):
    """Visualize path with vehicle positions every 30 meters."""
    
    x = np.array(path_dict['path_x'])
    y = np.array(path_dict['path_y'])
    s = np.array(path_dict['s_distances'])
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    
    # Plot the path line
    ax.plot(x, y, 'b-', linewidth=2, label='Reference Path', zorder=1, alpha=0.5)
    
    # Find vehicle positions every 30m
    interval = 30.0  # meters between vehicles
    vehicle_indices = [0]
    for i in range(1, len(s)):
        if s[i] - s[vehicle_indices[-1]] >= interval:
            vehicle_indices.append(i)
    
    print(f"\nVehicle positions: {len(vehicle_indices)} points (every {interval}m)")
    
    # Add vehicle position markers with heading
    car_width = 2.5  # size of car symbol (meters on plot)
    for idx in vehicle_indices:
        vx, vy = x[idx], y[idx]
        
        # Calculate heading at this point
        if idx > 0 and idx < len(x) - 1:
            dx = x[idx+1] - x[idx-1]
            dy = y[idx+1] - y[idx-1]
            heading = np.arctan2(dy, dx)
            
            # Draw vehicle as a small triangle/arrow
            head_len = car_width * 1.2
            
            # Vehicle body (triangle pointing in direction of travel)
            v_x = [vx,
                   vx - head_len * np.cos(heading - np.pi/6),
                   vx - head_len * np.cos(heading + np.pi/6)]
            v_y = [vy,
                   vy - head_len * np.sin(heading - np.pi/6),
                   vy - head_len * np.sin(heading + np.pi/6)]
            ax.fill(v_x, v_y, 'green', alpha=0.7, edgecolor='darkgreen', linewidth=1.5)
            
            # Add circle around vehicle
            circ = plt.Circle((vx, vy), car_width*0.8, color='none', 
                            edgecolor='blue', linewidth=1.5, alpha=0.6)
            ax.add_patch(circ)
    
    # Mark start/end point prominently
    if len(x) > 0:
        ax.plot(x[0], y[0], 'go', markersize=15, label='Start/End Point', zorder=5)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    save_path = Path.cwd() / "reference_path_with_vehicles.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    return save_path


def export_vehicle_markers(path_dict, interval=30.0):
    """Export vehicle positions and headings at specified interval.
    
    Returns:
        List of dicts with position (x,y), distance (s), and heading
    """
    x = np.array(path_dict['path_x'])
    y = np.array(path_dict['path_y'])
    s = np.array(path_dict['s_distances'])
    
    markers = []
    current_s = 0.0
    
    for i, dist in enumerate(s):
        if dist - current_s >= interval:
            current_s = dist
            
            if i > 0 and i < len(x) - 1:
                dx = x[i+1] - x[i-1]
                dy = y[i+1] - y[i-1]
                heading = np.arctan2(dy, dx)
                
                markers.append({
                    'index': int(i),
                    'distance_m': float(dist),
                    'x': float(x[i]),
                    'y': float(y[i]),
                    'heading_rad': float(heading),
                    'heading_deg': float(np.degrees(heading))
                })
    
    print(f"\nTotal vehicle markers: {len(markers)}")
    return markers


def main():
    """Main function."""
    import json
    
    print("\n" + "="*70)
    print("EXTRACTING CLOCKWISE REFERENCE LOOP")
    print("="*70)
    
    config = dict(
        use_render=False,
        num_scenarios=1,
        traffic_density=0,
        map_config=dict(lane_num=LANE_NUM, lane_width=LANE_WIDTH),
    )
    
    env = SimpleTrackEnv(config)
    
    try:
        env.reset(seed=0)
        
        # Extract path
        ref_path = extract_single_clockwise_loop(env)
        
        # Visualize with vehicle markers every 30m
        print("\n--- Generating visualization ---")
        visualize_with_vehicle_markers(ref_path, 
                           f"Reference Loop - Vehicle Positions Every {int(30)}m\n{len(ref_path['path_x'])} points, {ref_path['s_distances'][-1]:.1f}m")
        
        # Export vehicle markers to JSON
        print("\n--- Exporting vehicle markers ---")
        markers = export_vehicle_markers(ref_path, interval=30.0)
        
        output_data = {
            'metadata': {
                'total_path_length_m': float(ref_path['s_distances'][-1]),
                'num_points': len(ref_path['path_x']),
                'vehicle_marker_interval_m': 30.0,
                'num_vehicles': len(markers)
            },
            'reference_path': {
                'x': ref_path['path_x'],
                'y': ref_path['path_y'],
                's_distances': ref_path['s_distances']
            },
            'vehicle_markers': markers
        }
        
        json_path = Path.cwd() / "simple_track_reference_clean.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Saved JSON: {json_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("OUTPUT SUMMARY")
        print("="*70)
        print(f"Total path length: {ref_path['s_distances'][-1]:.1f} m")
        print(f"Reference path points: {len(ref_path['path_x'])}")
        print(f"Vehicle markers (every 30m): {len(markers)}")
        print(f"First marker at: s={markers[0]['distance_m']:.1f}m, pos=({markers[0]['x']:.2f}, {markers[0]['y']:.2f}), heading={markers[0]['heading_deg']:.1f}°")
        print(f"Last marker at: s={markers[-1]['distance_m']:.1f}m, pos=({markers[-1]['x']:.2f}, {markers[-1]['y']:.2f}), heading={markers[-1]['heading_deg']:.1f}°")
        print("="*70)
        
    finally:
        env.close()


if __name__ == "__main__":
    main()
