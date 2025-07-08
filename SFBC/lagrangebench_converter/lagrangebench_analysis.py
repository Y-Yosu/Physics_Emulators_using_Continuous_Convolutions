import os
import sys
import h5py
import numpy as np
import json
from typing import Dict, List, Any

def analyze_lagrangebench_structure():
    # Check for Lagrangebench data
    lagrange_path = "../../datasets/lagrangebench"
    
    if not os.path.exists(lagrange_path):
        print(f"Lagrangebench path not found: {lagrange_path}")
        return
    
    # Find .h5 files and organize by dataset
    datasets = {}
    for root, dirs, files in os.walk(lagrange_path):
        for file in files:
            if file.endswith('.h5'):
                filepath = os.path.join(root, file)
                # Extract dataset name from path
                rel_path = os.path.relpath(filepath, lagrange_path)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    dataset_name = parts[0]
                    if dataset_name not in datasets:
                        datasets[dataset_name] = {'files': [], 'metadata': None}
                    datasets[dataset_name]['files'].append(filepath)
    
    # Look for metadata.json files
    for dataset_name in datasets.keys():
        dataset_path = os.path.join(lagrange_path, dataset_name)
        metadata_path = os.path.join(dataset_path, 'metadata.json')
        if os.path.exists(metadata_path):
            datasets[dataset_name]['metadata'] = metadata_path
    
    print(f"Found {len(datasets)} datasets:")
    for dataset_name, dataset_info in datasets.items():
        print(f"  {dataset_name}: {len(dataset_info['files'])} files")
        if dataset_info['metadata']:
            print(f"    ✓ metadata.json found")
        else:
            print(f"    ✗ no metadata.json")
    
    for dataset_name, dataset_info in datasets.items():
        if 'tgv' in dataset_name.lower() or '2d_tgv' in dataset_name.lower():
            print(f"\n{'=' * 80}")
            print(f"TGV DATASET ANALYSIS: {dataset_name}")
            print(f"{'=' * 80}")
            
            # First analyze metadata if available
            if dataset_info['metadata']:
                analyze_metadata(dataset_info['metadata'])
            
            # Then analyze first file structure
            for filepath in sorted(dataset_info['files']):
                print(f"\n--- HDF5 FILE STRUCTURE: {os.path.basename(filepath)} ---")
                analyze_tgv_metadata(filepath)
                analyze_lagrangebench_file(filepath)
                break  # Only analyze first file in detail
    
    # Then analyze other 2D datasets
    for dataset_name, dataset_info in datasets.items():
        if ('2d' in dataset_name.lower() or '2D' in dataset_name) and 'tgv' not in dataset_name.lower():
            print(f"\n{'=' * 60}")
            print(f"DATASET: {dataset_name}")
            print(f"{'=' * 60}")
            
            # Analyze metadata if available
            if dataset_info['metadata']:
                analyze_metadata(dataset_info['metadata'])
            
            for filepath in sorted(dataset_info['files']):
                if any(split in filepath for split in ['train', 'valid', 'test']):
                    print(f"\n--- {os.path.basename(filepath)} ---")
                    analyze_lagrangebench_file(filepath)
                    break  # Only analyze first file for brevity

def analyze_metadata(metadata_path: str):
    print(f"\nMETADATA ANALYSIS: {os.path.basename(metadata_path)}")
    print("-" * 60)
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("PHYSICS PARAMETERS:")
        # Core simulation parameters
        core_params = ['case', 'solver', 'dim', 'dx', 'dt', 't_end', 'viscosity']
        for param in core_params:
            if param in metadata:
                print(f"  {param}: {metadata[param]}")
        
        # Derived parameters
        if 'dt' in metadata and 'write_every' in metadata:
            effective_dt = metadata['dt'] * metadata.get('write_every', 1)
            print(f"  effective_dt (dt * write_every): {effective_dt}")
        
        if 'sequence_length_train' in metadata and 't_end' in metadata:
            frames = metadata['sequence_length_train']
            total_time = metadata['t_end']
            frame_dt = total_time / (frames - 1) if frames > 1 else 0
            print(f"  frame_dt (t_end / (frames-1)): {frame_dt}")
        
        print("\nSPH PARAMETERS:")
        sph_params = ['default_connectivity_radius', 'artificial_alpha', 'density_evolution']
        for param in sph_params:
            if param in metadata:
                print(f"  {param}: {metadata[param]}")
        
        print("\nBOUNDARY CONDITIONS:")
        boundary_params = ['bounds', 'periodic_boundary_conditions', 'free_slip']
        for param in boundary_params:
            if param in metadata:
                print(f"  {param}: {metadata[param]}")
        
        print("\nFORCES:")
        force_params = ['g_ext_magnitude', 'p_bg_factor']
        for param in force_params:
            if param in metadata:
                print(f"  {param}: {metadata[param]}")
        
        print("\nDATA STATISTICS:")
        stats_params = ['sequence_length_train', 'num_trajs_train', 'sequence_length_test', 
                       'num_trajs_test', 'num_particles_max', 'write_every']
        for param in stats_params:
            if param in metadata:
                print(f"  {param}: {metadata[param]}")
        
        # Velocity and acceleration statistics
        if 'vel_mean' in metadata and 'vel_std' in metadata:
            print(f"\nVELOCITY STATISTICS:")
            print(f"  mean: {metadata['vel_mean']}")
            print(f"  std: {metadata['vel_std']}")
        
        if 'acc_mean' in metadata and 'acc_std' in metadata:
            print(f"\nACCELERATION STATISTICS:")
            print(f"  mean: {metadata['acc_mean']}")
            print(f"  std: {metadata['acc_std']}")
            
    except Exception as e:
        print(f"Error reading metadata: {e}")

def analyze_tgv_metadata(filepath: str):    
    print(f"TGV DATASET COMPLETE STRUCTURE ANALYSIS")
    print(f"File: {filepath}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"\nFILE-LEVEL ATTRIBUTES:")
            if f.attrs:
                for attr_name, attr_value in f.attrs.items():
                    print(f"  {attr_name}: {attr_value} (type: {type(attr_value).__name__})")
            else:
                print("  No file-level attributes found")
            
            print(f"\nROOT LEVEL KEYS:")
            root_keys = list(f.keys())
            print(f"  Total trajectories: {len(root_keys)}")
            print(f"  Trajectory IDs: {root_keys[:5]}{'...' if len(root_keys) > 5 else ''}")
            
            # Analyze first trajectory in detail
            if root_keys:
                first_traj_id = root_keys[0]
                print(f"\nTRAJECTORY '{first_traj_id}' DETAILED ANALYSIS:")
                first_traj = f[first_traj_id]
                
                print(f"  Object type: {type(first_traj).__name__}")
                
                # Check trajectory-level attributes
                if hasattr(first_traj, 'attrs') and first_traj.attrs:
                    print(f"  Trajectory attributes:")
                    for attr_name, attr_value in first_traj.attrs.items():
                        print(f"    {attr_name}: {attr_value} (type: {type(attr_value).__name__})")
                else:
                    print("  No trajectory-level attributes")
                
                # Check what datasets/groups are in the trajectory
                if isinstance(first_traj, h5py.Group):
                    traj_keys = list(first_traj.keys())
                    print(f"  Contains datasets: {traj_keys}")
                    
                    # Analyze each dataset in detail
                    for key in traj_keys:
                        print(f"\n  DATASET '{key}':")
                        dataset = first_traj[key]
                        print(f"    Type: {type(dataset).__name__}")
                        
                        if isinstance(dataset, h5py.Dataset):
                            print(f"    Shape: {dataset.shape}")
                            print(f"    Dtype: {dataset.dtype}")
                            
                            # Show dataset attributes
                            if dataset.attrs:
                                print(f"    Attributes:")
                                for attr_name, attr_value in dataset.attrs.items():
                                    print(f"      {attr_name}: {attr_value} (type: {type(attr_value).__name__})")
                            else:
                                print(f"    No dataset attributes")
                            
                            # Show sample data for small datasets
                            if dataset.size is not None and dataset.size < 100:
                                print(f"    Sample data: {dataset[:]}")
                            elif len(dataset.shape) > 0 and dataset.shape[0] > 0:
                                print(f"    First few values: {dataset[:min(3, dataset.shape[0])]}")
            
            # Recursively find ALL attributes in the file
            print(f"\nALL ATTRIBUTES IN FILE (COMPREHENSIVE SEARCH):")
            find_all_attributes(f, "")
                    
    except Exception as e:
        print(f"Error analyzing TGV file: {e}")
        import traceback
        traceback.print_exc()

def find_all_attributes(obj, path=""):
    if hasattr(obj, 'attrs') and obj.attrs:
        for attr_name, attr_value in obj.attrs.items():
            print(f"  {path}/{attr_name}: {attr_value}")
    
    # Recursively check children
    if hasattr(obj, 'keys'):
        for key in obj.keys():
            child_path = f"{path}/{key}" if path else key
            try:
                child_obj = obj[key]
                find_all_attributes(child_obj, child_path)
            except Exception as e:
                print(f"  {child_path}: <unable to access - {e}>")

def analyze_lagrangebench_file(filepath: str):   
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"File: {os.path.basename(filepath)}")
            print(f"Path: {os.path.relpath(filepath, '..')}")
            print(f"Root keys: {list(f.keys())}")
            
            # Count total trajectories
            trajectory_count = len(f.keys())
            print(f"Total trajectories: {trajectory_count}")
            
            # Analyze first few trajectories
            for idx, traj_id in enumerate(list(f.keys())[:2]):  # First 2 trajectories
                print(f"\n  Trajectory {idx+1}: ID='{traj_id}'")
                traj_group = f[traj_id]
                
                if isinstance(traj_group, h5py.Group):
                    data_keys = list(traj_group.keys())
                    print(f"    Data keys: {data_keys}")
                    
                    # Analyze position data
                    if 'position' in traj_group:
                        positions = traj_group['position']
                        if isinstance(positions, h5py.Dataset):
                            print(f"    position: shape={positions.shape}, dtype={positions.dtype}")
                            print(f"      Timesteps: {positions.shape[0]}")
                            print(f"      Particles: {positions.shape[1]}")
                            print(f"      Dimensions: {positions.shape[2]}")
                            
                            # Sample positions
                            pos_array = np.array(positions)
                            print(f"      First timestep, first 3 particles:")
                            print(f"        {pos_array[0, :3, :]}")
                            print(f"      Position ranges:")
                            for dim in range(positions.shape[2]):
                                print(f"        Dim {dim}: [{pos_array[:, :, dim].min():.3f}, {pos_array[:, :, dim].max():.3f}]")
                    
                    # Analyze particle type data
                    if 'particle_type' in traj_group:
                        particle_types = traj_group['particle_type']
                        if isinstance(particle_types, h5py.Dataset):
                            print(f"    particle_type: shape={particle_types.shape}, dtype={particle_types.dtype}")
                            
                            # Get unique particle types
                            pt_array = np.array(particle_types)
                            unique_types, counts = np.unique(pt_array, return_counts=True)
                            print(f"      Unique types: {unique_types}")
                            print(f"      Type counts: {dict(zip(unique_types, counts))}")
                    
                    # Check for any other data
                    other_keys = [k for k in data_keys if k not in ['position', 'particle_type']]
                    if other_keys:
                        print(f"    Other data keys: {other_keys}")
            
            # Get dataset type from filepath
            dataset_name = identify_dataset_type(filepath)
            if dataset_name:
                print(f"\n  Dataset type: {dataset_name}")
            
            print(f"\nFile summary:")
            print(f"  Trajectories: {trajectory_count}")
            if trajectory_count > 0 and '00000' in f:
                sample_traj = f['00000']
                if isinstance(sample_traj, h5py.Group) and 'position' in sample_traj:
                    pos_data = sample_traj['position']
                    if isinstance(pos_data, h5py.Dataset):
                        pos_shape = pos_data.shape
                        print(f"  Timesteps per trajectory: {pos_shape[0]}")
                        print(f"  Particles: {pos_shape[1]}")
                        print(f"  Spatial dimensions: {pos_shape[2]}")
                    
    except Exception as e:
        print(f"Failed to analyze {filepath}: {e}")
        import traceback
        traceback.print_exc()

def identify_dataset_type(filepath: str) -> str:
    path_lower = filepath.lower()
    
    # Check for dataset identifiers
    if 'tgv' in path_lower or 'taylor' in path_lower:
        return "2D TGV (Taylor-Green Vortex)"
    elif 'rpf' in path_lower or 'poiseuille' in path_lower:
        return "2D RPF (Reverse Poiseuille Flow)"
    elif 'ldc' in path_lower or 'cavity' in path_lower:
        return "2D LDC (Lid-Driven Cavity)"
    elif 'dam' in path_lower:
        return "2D DAM (Dam Break)"
    
    return ""



def main():
    print("LAGRANGEBENCH METADATA ANALYSIS")
    print("=" * 80)
    analyze_lagrangebench_structure()
    
if __name__ == "__main__":
    main() 