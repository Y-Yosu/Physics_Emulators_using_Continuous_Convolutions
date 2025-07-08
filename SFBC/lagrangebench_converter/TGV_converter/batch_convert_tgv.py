import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
LAGRANGEBENCH_TGV_PATH = "../../../datasets/lagrangebench/2D_TGV_2500_10kevery100"
OUTPUT_BASE_PATH = "../../../datasets/SFBC/TGV_SFBC_dataset/dataset"
CONVERTER_SCRIPT = "lagrangebench_tgv_to_sfbc_converter.py"
METADATA_PATH = os.path.join(LAGRANGEBENCH_TGV_PATH, "metadata.json")

# Dataset splits configuration
DATASET_CONFIG = {
    'train': {
        'input_file': 'train.h5',
        'num_trajectories': 100,
        'output_dir': 'train'
    },
    'test': {
        'input_file': 'test.h5',
        'num_trajectories': 50,
        'output_dir': 'test'
    },
    'valid': {
        'input_file': 'valid.h5',
        'num_trajectories': 50,
        'output_dir': 'valid'
    }
}

def convert_trajectory(split_name, trajectory_id, input_file, output_file, metadata_path):    
    cmd = [
        'python', CONVERTER_SCRIPT,
        input_file,
        output_file,
        '--trajectory', trajectory_id,
        '--metadata', metadata_path
    ]
    
    print(f"Converting {split_name} trajectory {trajectory_id}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully converted {trajectory_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert {trajectory_id}")
        print(f"Error output: {e.stderr}")
        return False

def batch_convert_split(split_name, config):
    print(f"\n{'='*70}")
    print(f"CONVERTING {split_name.upper()} SPLIT")
    print(f"{'='*70}")
    
    input_file = os.path.join(LAGRANGEBENCH_TGV_PATH, config['input_file'])
    output_dir = os.path.join(OUTPUT_BASE_PATH, config['output_dir'])
    
    # Verify input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata: {METADATA_PATH}")
    
    success_count = 0
    total_count = config['num_trajectories']
    
    for i in range(total_count):
        trajectory_id = f"{i:05d}"  # Zero-padded trajectory ID
        
        # Use SFBC-style naming convention
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(output_dir, f"tgv_simulation_{trajectory_id}_{timestamp}.hdf5")
        
        if convert_trajectory(split_name, trajectory_id, input_file, output_file, METADATA_PATH):
            success_count += 1
        
        # Progress update every 10 conversions
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{total_count} trajectories processed")
    
    print(f"\n{split_name.upper()} SPLIT COMPLETE:")
    print(f"  Successfully converted: {success_count}/{total_count} trajectories")
    print(f"  Success rate: {success_count/total_count*100:.1f}%")
    print(f"  Output directory: {output_dir}")
    
    return success_count == total_count

def verify_environment():
    print("🔍 VERIFYING ENVIRONMENT:")
    
    # Check converter script
    if not os.path.exists(CONVERTER_SCRIPT):
        print(f"Converter script not found: {CONVERTER_SCRIPT}")
        print("Make sure you're running from the TGV_converter directory")
        return False
    print(f"Converter script found: {CONVERTER_SCRIPT}")
    
    # Check Lagrangebench TGV directory
    if not os.path.exists(LAGRANGEBENCH_TGV_PATH):
        print(f"Lagrangebench TGV path not found: {LAGRANGEBENCH_TGV_PATH}")
        print("Please check the dataset path")
        return False
    print(f"TGV dataset path found: {LAGRANGEBENCH_TGV_PATH}")
    
    # Check metadata file
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata file not found: {METADATA_PATH}")
        print("Please check if metadata.json exists in the dataset directory")
        return False
    print(f"Metadata file found: {METADATA_PATH}")
    
    # Check for input files
    missing_files = []
    for split_name, config in DATASET_CONFIG.items():
        input_file = os.path.join(LAGRANGEBENCH_TGV_PATH, config['input_file'])
        if not os.path.exists(input_file):
            missing_files.append(input_file)
        else:
            print(f"{split_name} file found: {config['input_file']}")
    
    if missing_files:
        print(f"Missing input files:")
        for f in missing_files:
            print(f"  {f}")
        return False
    
    print("Environment verification passed")
    return True

def cleanup_old_files():    
    print("CLEANING UP OLD FILES...")
    
    for split_name, config in DATASET_CONFIG.items():
        output_dir = os.path.join(OUTPUT_BASE_PATH, config['output_dir'])
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
            print(f"Removed old {split_name} directory")
    
    print("Cleanup complete")

def display_metadata_info():
    print("\nMETADATA INFORMATION:")
    try:
        import json
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        print(f"  Case: {metadata.get('case', 'Unknown')}")
        print(f"  Solver: {metadata.get('solver', 'Unknown')}")
        print(f"  Dimensions: {metadata.get('dim', 'Unknown')}")
        print(f"  Simulation dt: {metadata.get('dt', 'Unknown')}")
        print(f"  Write every: {metadata.get('write_every', 'Unknown')} steps")
        
        # Calculate effective dt
        if 'dt' in metadata and 'write_every' in metadata:
            effective_dt = metadata['dt'] * metadata['write_every']
            print(f"  Effective dt: {effective_dt}")
        
        print(f"  Particle spacing: {metadata.get('dx', 'Unknown')}")
        print(f"  Support radius: {metadata.get('default_connectivity_radius', 'Unknown')}")
        print(f"  Viscosity: {metadata.get('viscosity', 'Unknown')}")
        print(f"  Domain bounds: {metadata.get('bounds', 'Unknown')}")
        print(f"  Periodic BC: {metadata.get('periodic_boundary_conditions', 'Unknown')}")
        print(f"  Final time: {metadata.get('t_end', 'Unknown')}")
        print(f"  Sequence length: {metadata.get('sequence_length_train', 'Unknown')}")
        print(f"  Max particles: {metadata.get('num_particles_max', 'Unknown')}")
        
    except Exception as e:
        print(f"Error reading metadata: {e}")

def main():
    print("="*80)
    print("TGV BATCH CONVERTER WITH METADATA")
    print("="*80)
    
    print(f"Input path: {LAGRANGEBENCH_TGV_PATH}")
    print(f"Output path: {OUTPUT_BASE_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Using converter: {CONVERTER_SCRIPT}")
    
    # Display metadata information
    display_metadata_info()
    
    # Verify environment
    if not verify_environment():
        print("Environment verification failed. Aborting.")
        return False
    
    # Ask user about cleanup
    response = input("\nDo you want to remove old converted files? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        cleanup_old_files()
    
    # Start conversion
    print("\nSTARTING BATCH CONVERSION...")
    total_success = True
    overall_stats = {'total': 0, 'success': 0}
    start_time = datetime.now()
    
    # Convert each split
    for split_name, config in DATASET_CONFIG.items():
        print(f"\nStarting {split_name} conversion...")
        
        split_success = batch_convert_split(split_name, config)
        total_success &= split_success
        
        overall_stats['total'] += config['num_trajectories']
        if split_success:
            overall_stats['success'] += config['num_trajectories']
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\nFINAL SUMMARY:")
    print(f"  Total trajectories: {overall_stats['total']}")
    print(f"  Successfully converted: {overall_stats['success']}")
    print(f"  Failed conversions: {overall_stats['total'] - overall_stats['success']}")
    print(f"  Success rate: {overall_stats['success']/overall_stats['total']*100:.1f}%")
    print(f"  Total time: {duration}")
    
    if total_success:
        print(f"\nALL CONVERSIONS COMPLETED SUCCESSFULLY!")
        print(f"  {OUTPUT_BASE_PATH}/train/    - 100 training simulations")
        print(f"  {OUTPUT_BASE_PATH}/test/     - 50 test simulations") 
        print(f"  {OUTPUT_BASE_PATH}/valid/    - 50 validation simulations")
        print(f"  Total converted files: {overall_stats['success']}")
    else:
        print("\nSome conversions failed. Check error messages above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 