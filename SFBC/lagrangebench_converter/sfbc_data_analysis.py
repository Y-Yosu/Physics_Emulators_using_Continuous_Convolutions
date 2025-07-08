import os
import sys
import h5py
import numpy as np
from typing import Dict, List, Any, Union

# Add SFBC modules to path
sys.path.append('/home/yusuf/Physics_Emulators_using_Continuous_Convolutions/SFBC')

try:
    from BasisConvolution.util.dataloader import processFolder, datasetLoader
    from BasisConvolution.util.datautils import parseFile
    print("Successfully imported SFBC modules")
except ImportError as e:
    print(f"Failed to import SFBC modules: {e}")
    sys.exit(1)

def analyze_dataset_directories():
    datasets_path = "../datasets/SFBC"
    
    print("=" * 80)
    print("COMPREHENSIVE SFBC DATASET II ANALYSIS")
    print("=" * 80)
    
    if not os.path.exists(datasets_path):
        print(f"Dataset path not found: {datasets_path}")
        return []
    
    # Focus on SFBC_dataset_II
    dataset_ii_path = os.path.join(datasets_path, "SFBC_dataset_II/dataset")
    
    if not os.path.exists(dataset_ii_path):
        print(f"SFBC_dataset_II not found at: {dataset_ii_path}")
        return []
    
    # Analyze train and test folders
    dataset_dirs = []
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_ii_path, split)
        if os.path.exists(split_path):
            hdf5_files = [f for f in os.listdir(split_path) if f.endswith('.hdf5')]
            if hdf5_files:
                dataset_dirs.append((split_path, len(hdf5_files), split))
    
    print(f"Found SFBC_dataset_II structure:")
    for i, (path, count, split) in enumerate(dataset_dirs):
        print(f"  {split}: {count} .hdf5 files")
    
    return dataset_dirs

def analyze_with_sfbc_dataloader(dataset_path: str, split_name: str):
    print(f"\nAnalyzing SFBC_dataset_II {split_name} set: {os.path.relpath(dataset_path, '../datasets/SFBC')}")
    
    # List files in the directory
    hdf5_files = [f for f in os.listdir(dataset_path) if f.endswith('.hdf5')]
    print(f"Files in {split_name} set: {len(hdf5_files)}")
    for i, filename in enumerate(sorted(hdf5_files)[:5]):  # Show first 5
        print(f"  {i+1}: {filename}")
    if len(hdf5_files) > 5:
        print(f"  {len(hdf5_files) - 5} more files")
    
    # SFBC hyperparameters
    hyperparams = {
        'frameDistance': 1,
        'dataDistance': 1,
        'maxUnroll': 0,
        'historyLength': 1,
        'zeroOffset': True,
        'dataLimit': 0,
        'verbose': True,
        'batch_size': 1
    }
    
    print(f"\nUsing hyperparameters: {hyperparams}")
    
    try:
        processed_files = processFolder(hyperparams, dataset_path)
        print(f"Returned {len(processed_files)} processed file(s)")
        
        if not processed_files:
            print("No files processed")
            return
        
        # Analyze each processed file (limit to first 3 for brevity)
        for i, file_data in enumerate(processed_files[:3]):
            print(f"\n{'=' * 60}")
            print(f"PROCESSED FILE {i+1} (of {len(processed_files)})")
            print(f"{'=' * 60}")
            
            print(f"File data keys: {list(file_data.keys())}")
            
            # Show key information
            for key in ['fileName', 'isTemporalData', 'style']:
                if key in file_data:
                    print(f"  {key}: {file_data[key]}")
            
            # Show sample info more concisely
            if 'samples' in file_data:
                samples = file_data['samples']
                print(f"  samples: {len(samples)} total")
                if len(samples) > 0:
                    print(f"    First 5: {samples[:5]}")
                    print(f"    Last 5: {samples[-5:]}")
                    print(f"    Sample range: {min(samples) if samples else 'N/A'} to {max(samples) if samples else 'N/A'}")
            
            if 'frames' in file_data:
                frames = file_data['frames']
                print(f"  frames: {len(frames)} total")
                if len(frames) > 0:
                    print(f"    First 5: {frames[:5]}")
                    print(f"    Last 5: {frames[-5:]}")
                    print(f"    Frame range: {min(frames) if frames else 'N/A'} to {max(frames) if frames else 'N/A'}")
            
            # Show other metadata
            for key in ['frameDistance', 'frameSpacing', 'maxRollout', 'skip', 'limit']:
                if key in file_data:
                    print(f"  {key}: {file_data[key]}")
        
        if len(processed_files) > 3:
            print(f"\n... and {len(processed_files) - 3} more processed files")
        
        # Test dataloader
        print(f"\n{'=' * 60}")
        print("CREATING SFBC DATALOADER")
        print(f"{'=' * 60}")
        
        try:
            loader = datasetLoader(processed_files)
            print(f"Created datasetLoader with {len(loader)} samples")
            
            print(f"\nTesting dataloader __getitem__ (first 3 samples)...")
            
            # Test first few samples
            for i in range(min(3, len(loader))):
                print(f"\n--- Sample {i} ---")
                try:
                    result = loader[i]
                    print(f"Result type: {type(result)}")
                    print(f"Result length: {len(result)}")
                    
                    # Handle different tuple structures
                    if isinstance(result, tuple) and len(result) >= 3:
                        print(f"  Tuple contents:")
                        for j, item in enumerate(result):
                            if isinstance(item, dict):
                                print(f"    [{j}]: Dict with keys: {list(item.keys())}")
                            else:
                                print(f"    [{j}]: {type(item).__name__} = {item}")
                    else:
                        print(f"  Result: {result}")
                except Exception as e:
                    print(f"  Error getting sample {i}: {e}")
                    
        except Exception as e:
            print(f"Failed to create dataloader: {e}")
        
        if processed_files:
            first_file = processed_files[0]
            if 'fileName' in first_file:
                inspect_hdf5_data(first_file['fileName'], detailed=True)
                
    except Exception as e:
        print(f"processFolder failed: {e}")
        import traceback
        traceback.print_exc()

def inspect_hdf5_data(filename: str, detailed: bool = False):
    print(f"\nDetailed HDF5 Analysis: {os.path.basename(filename)}")
    
    try:
        with h5py.File(filename, 'r') as f:
            print(f"Root keys: {list(f.keys())}")
            
            # Look for simulationExport (the actual data location)
            for root_key in f.keys():
                if root_key == 'simulationExport':
                    print(f"\nExploring {root_key}:")
                    group = f[root_key]
                    if isinstance(group, h5py.Group):
                        timestep_keys = list(group.keys())
                        print(f"  Timestep IDs: {len(timestep_keys)} total")
                        if len(timestep_keys) > 0:
                            print(f"    First 5: {timestep_keys[:5]}")
                            print(f"    Last 5: {timestep_keys[-5:]}")
                            
                            # Check if these are timestep indices
                            try:
                                # Try to parse as zero-padded numbers
                                numeric_keys = []
                                for k in timestep_keys:
                                    if k.isdigit():
                                        numeric_keys.append(int(k))
                                
                                if numeric_keys:
                                    print(f"    Numeric range: {min(numeric_keys)} to {max(numeric_keys)}")
                                    print(f"    Total timesteps: {len(numeric_keys)}")
                                else:
                                    print(f"    Keys are not numeric: {timestep_keys[:10]}")
                            except:
                                print(f"    Could not parse timestep keys as numbers")
                            
                            # Inspect first few timesteps
                            for idx, timestep_id in enumerate(timestep_keys[:2]):  # First 2 timesteps
                                print(f"\n  Timestep {idx+1}: '{timestep_id}'")
                                timestep_group = group[timestep_id]
                                
                                if isinstance(timestep_group, h5py.Group):
                                    data_keys = list(timestep_group.keys())
                                    print(f"    Data keys: {data_keys}")
                                    
                                    # Count datasets vs groups
                                    datasets = {}
                                    groups = {}
                                    
                                    for key in data_keys:
                                        try:
                                            data = timestep_group[key]
                                            if isinstance(data, h5py.Dataset):
                                                datasets[key] = data
                                                print(f"      {key}: Dataset shape={data.shape}, dtype={data.dtype}")
                                            elif isinstance(data, h5py.Group):
                                                groups[key] = data
                                                print(f"      {key}: Group with {len(data.keys())} items")
                                        except Exception as e:
                                            print(f"      {key}: Error reading - {e}")
                                    
                                    # Show particle count consistency
                                    particle_counts = []
                                    for key, dataset in datasets.items():
                                        if len(dataset.shape) > 0:
                                            particle_counts.append(dataset.shape[0])
                                    
                                    if particle_counts:
                                        unique_counts = list(set(particle_counts))
                                        if len(unique_counts) == 1:
                                            print(f"    All datasets consistent with {unique_counts[0]} particles")
                                        else:
                                            print(f"    Inconsistent particle counts: {unique_counts}")
                                    
                                    # Sample actual data from first timestep only
                                    if detailed and idx == 0 and datasets:
                                        print(f"\n    PARTICLE DATA SAMPLES (First Timestep):")
                                        print(f"    " + "-" * 40)
                                        
                                        # Show position data
                                        if 'x' in datasets:
                                            positions = np.array(datasets['x'][:3])  # First 3 particles
                                            print(f"    Positions (x): shape={datasets['x'].shape}")
                                            print(f"      First 3 particles: {positions}")
                                            print(f"      Position range: X[{datasets['x'][:].min():.3f}, {datasets['x'][:].max():.3f}]")
                                            
                                        # Show other particle properties
                                        for prop in ['rho', 'vols', 'jitter', 'u', 'v', 'p']:
                                            if prop in datasets:
                                                data = np.array(datasets[prop])
                                                print(f"    {prop}: shape={datasets[prop].shape}, range=[{data.min():.6f}, {data.max():.6f}]")
                                        
                                        # Show gradient properties
                                        grad_props = [k for k in datasets.keys() if 'grad' in k.lower()]
                                        if grad_props:
                                            print(f"    Gradient properties: {grad_props}")
                                            for prop in grad_props[:3]:  # First 3 gradient properties
                                                data = np.array(datasets[prop])
                                                print(f"      {prop}: shape={datasets[prop].shape}, range=[{data.min():.6f}, {data.max():.6f}]")
                                
                                if idx >= 1:  # Only show first 2 timesteps
                                    break
                
                elif root_key == 'boundaryInformation':
                    print(f"\nExploring {root_key}:")
                    boundary_group = f[root_key]
                    if isinstance(boundary_group, h5py.Group):
                        boundary_keys = list(boundary_group.keys())
                        print(f"  Boundary keys: {boundary_keys}")
                        
                        # Show boundary data
                        for key in boundary_keys[:3]:  # First 3 boundary items
                            try:
                                item = boundary_group[key]
                                if isinstance(item, h5py.Dataset):
                                    print(f"    {key}: Dataset shape={item.shape}, dtype={item.dtype}")
                                elif isinstance(item, h5py.Group):
                                    print(f"    {key}: Group with {len(item.keys())} items")
                            except Exception as e:
                                print(f"    {key}: Error reading - {e}")
                                
                # Look for other root keys
                elif root_key not in ['simulationExport', 'boundaryInformation']:
                    print(f"\nOther root key '{root_key}':")
                    try:
                        item = f[root_key]
                        if isinstance(item, h5py.Dataset):
                            print(f"  Dataset: shape={item.shape}, dtype={item.dtype}")
                        elif isinstance(item, h5py.Group):
                            print(f"  Group with {len(item.keys())} items")
                    except Exception as e:
                        print(f"  Error reading: {e}")
            
            # File structure recap
            print(f"\nFile structure recap:")
            print(f"  Root keys: {list(f.keys())}")
            
            # Get total file size info
            if 'simulationExport' in f:
                export_group = f['simulationExport']
                if isinstance(export_group, h5py.Group):
                    print(f"  simulationExport: {len(export_group.keys())} timesteps")
                    
    except Exception as e:
        print(f"Failed to inspect {filename}: {e}")

def main():
    """Main analysis function"""
    dataset_dirs = analyze_dataset_directories()
    
    if not dataset_dirs:
        print("No SFBC_dataset_II directories found")
        return
    
    # Analyze test set only for now (to keep output manageable)
    for dataset_path, file_count, split_name in dataset_dirs:
        if split_name == 'test':  # Only test set
            print(f"\n{'=' * 80}")
            print(f"ANALYZING {split_name.upper()} SET")
            print(f"{'=' * 80}")
            
            analyze_with_sfbc_dataloader(dataset_path, split_name)
            break
        
if __name__ == "__main__":
    main() 